# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import inspect

from loggers import Timer, timer
from utils.text import format_text
from utils.callbacks import apply_callbacks
from .prompts import add_prompt_wrapper
from .base_language_model import BaseLanguageModel
from .tools import execute_code, extract_code, format_code_result
from .conversations import Chat, Conversation, Message, get_message_selector

logger = logging.getLogger(__name__)

class TextGenerator(BaseLanguageModel):
    _directories    = {
        ** BaseLanguageModel._directories,
        'vectors_dir'   : '{root}/{self.name}/vectors',
        'conv_dir'      : '{root}/{self.name}/conversations'
    }

    _default_pretrained_model   = None
    
    _default_loss   = 'TextLoss'
    _default_metrics    = 'TextAccuracy'
    
    output_signature    = BaseLanguageModel.text_signature
    
    def __init__(self, * args, ** kwargs):
        super().__init__(* args, ** kwargs)
        
        self._chats = {}
    
    @property
    def max_output_length(self):
        return getattr(self.model, 'max_output_length', None)
    
    def get_chat(self, chat_id):
        if chat_id not in self._chats:
            filename = os.path.join(self.conv_dir, '{}.json'.format(chat_id))
            if os.path.exists(filename):
                self._chats[chat_id] = Chat.load(filename)
            else:
                self._chats[chat_id] = Chat(id = chat_id)
        
        return self._chats[chat_id]
    
    def get_conv(self, chat_id, ** kwargs):
        chat = self.get_chat(chat_id)
        return chat, chat.get_conv(** kwargs)
    
    def get_messages(self, chat_id, *, chat = None, conv = None, message_selector = 'last', ** kwargs):
        if conv is None: chat, conv = self.get_conv(chat_id, ** kwargs)
        if conv is None: return (None, None, [])
        
        selector = get_message_selector(message_selector)
        return chat, conv, selector.get_messages(
            conv = conv, chat = chat, tokenizer = self.tokenizer, ** kwargs
        )
    
    @timer
    @add_prompt_wrapper('default')
    def infer(self,
              text,
              *,
              
              format    = None,
              
              chat  = None,
              conv  = None,
              chat_id   = 'default',
              conv_id   = None,
              new_conv  = False,
              messages  = None,
              message_selector  = 'last',
              max_input_length  = 4096,
              
              tools = None,
              max_depth = 5,
              _depth    = 0,
              
              possible_answers  = None,
              
              stop_words  = None,
              max_new_tokens  = 2048,
              add_answer_start    = True,

              stream_text   = False,
              stream_callback   = None,

              request_id    = None,
              wait_finalization = False,
              _initial_conv_state   = None,

              callbacks = None,
              predicted = None,

              ** kwargs
             ):
        if hasattr(stream_callback, 'build'): stream_callback.build()
        
        query = text
        if format: text = format_text(format, text = text, ** kwargs)
        
        if messages is None:
            chat, conv, messages = self.get_messages(
                conv    = conv,
                chat_id = chat_id,
                conv_id = conv_id,
                new_conv    = new_conv,
                max_length  = max_input_length,
                message_selector    = message_selector,
                ** kwargs
            )
        
        if _depth == 0:
            if conv is not None and _initial_conv_state is None:
                _initial_conv_state = conv.get_state()
        
            if tools and self.runtime == 'trt_llm':
                self.model.engine.logits_processor_map['tool_stopper'].tokenizer = self.tokenizer
                if 'tool_stopper' not in kwargs.get('logits_processor_names', []):
                    kwargs.setdefault('logits_processor_names', []).append('tool_stopper')
        
        if possible_answers:
            kwargs['allowed_tokens'] = pad_batch(
                self.encode_text(
                    possible_answers, add_sos_and_eos = False, return_type = 'np'
                ),
                pad_value = self.blank_token_idx,
                dtype = 'int32'
            )

        prompt, tokens = self.get_input(
            text,
            messages    = messages,
            pinned_messages = conv.pinned if conv is not None else [],
            python_tools    = tools,

            max_length  = max_input_length,
            return_text     = True,
            
            ** kwargs
        )
        
        if hasattr(stream_callback, 'is_stopped') and stream_callback.is_stopped(request_id):
            if _depth == 0: stream_callback.pop(request_id)
            return {}
        
        out = self.compiled_infer(
            tokens[None],
            
            max_new_tokens  = max_new_tokens,
            
            request_id  = request_id,
            decode_fn   = self.decode_output if stream_text else None,
            stream_callback = stream_callback,
            
            ** kwargs
        )
        
        pred = self.decode_output(out)[0]
        if isinstance(pred, list) and len(pred) == 1: pred = pred[0]
        if add_answer_start and kwargs.get('answer_start', None):
            pred = kwargs['answer_start'] + pred

        _abort = (
            hasattr(stream_callback, 'is_stopped') and stream_callback.is_stopped(request_id)
        )

        if not _abort:
            if conv is not None:
                if text:
                    conv.append(
                        text, role = 'user', query = query, format = format, ** kwargs
                    )
                conv.append(
                    pred, role = 'assistant', prompt = prompt
                )
            
            if tools:
                tool_code = extract_code(pred)
                if len(tool_code) == 1: tool_code = tool_code[0]
                
                if tool_code and 'print(' in tool_code or any(tool.name + '(' in tool_code for tool in tools):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('Tool call detected :\n{}'.format(tool_code))
            
                    tool_result  = execute_code(
                        tool_code, tools = tools, conv = conv, model = self
                    )
                    tool_message = format_code_result(tool_result)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('Tools call result : {}'.format(tool_message))

                    if tool_message:
                        conv.append(
                            tool_message,
                            role    = 'tool_result',
                            code    = tool_code,
                            result  = tool_result
                        )

                        return self.infer(
                            None,
                            chat    = chat,
                            conv    = conv,
                            message_selector    = message_selector,
                            max_input_length    = max_input_length,

                            request_id  = request_id,
                            stream_text = stream_text,
                            stream_callback = stream_callback,
                            _initial_conv_state = _initial_conv_state,

                            tools   = tools if _depth + 1 < max_depth else None,
                            max_depth   = max_depth,
                            _depth  = _depth + 1,
                            _initial_message_idx    = _initial_message_idx,

                            stop_words  = None,
                            max_new_tokens  = max_new_tokens,
                            add_answer_start    = add_answer_start,

                            _add_prompts    = False,

                            ** kwargs
                        )

        result = {
            'predicted' : pred,
            
            'query' : query,
            'format'    : format,
            'prompt'    : prompt,
            'input_tokens'  : tokens,
            ** kwargs
        }

        if _depth > 0: return result
        
        if stream_callback is not None:
            if request_id is not None:
                if wait_finalization:
                    _abort = not stream_callback.wait_finalize(request_id)

                stream_callback((request_id, inspect._empty))
                stream_callback({'id' : request_id, 'type' : 'status', 'content' : 'finished'})

            else:
                if not callable(stream_callback): stream_callback = stream_callback.put
                stream_callback(inspect._empty)

        if _abort and _initial_conv_state is not None:
            logger.info('Resetting conv to its initial state')
            conv.set_state(_initial_conv_state)

        if callbacks:
            apply_callbacks(callbacks, {}, result, save = False)

        if request_id is not None:
            logger.info('Request {} {} !'.format(
                request_id, 'finished' if not _abort else 'aborted'
            ))
        
        return result

    answer  = add_prompt_wrapper('answer', fn = infer)
    
    translate   = add_prompt_wrapper('translate',   fn = answer)
    reformulate = add_prompt_wrapper('reformulate', fn = answer)
    describe    = add_prompt_wrapper('describe',    fn = answer)
    summarize   = add_prompt_wrapper('summarize',   fn = answer)
    extract_entities    = add_prompt_wrapper('extract_entities', fn = answer)
