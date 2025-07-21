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
import re
import logging
import inspect

from functools import partial

from loggers import Timer, timer
from utils.text import format_text, parse_document, search_on_web
from utils.callbacks import apply_callbacks
from .inference_manager import InferenceManager
from .prompts import add_prompt_wrapper, get_translation
from .base_language_model import BaseLanguageModel
from .tools import execute_code, extract_code, format_code_result, normalize_tools, remove_simulated_output
from .conversations import Chat, Conversation, Message, get_message_selector, get_messages

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
    
    def get_conv(self, chat_id = 'default', chat = None, ** kwargs):
        if chat is None: chat = self.get_chat(chat_id)
        return chat.get_conv(** kwargs)
    
    @timer
    @add_prompt_wrapper('default')
    def infer(self,
              text,
              *,
              
              prefix    = None,
              format    = None,
              
              conv  = None,
              conv_id   = None,
              messages  = None,
              message_selector  = 'last',
              max_input_length  = None,
              
              tools = None,
              max_depth = 5,
              allow_code_execution  = False,
              
              stop_words  = None,
              max_new_tokens  = 2048,
              possible_answers  = None,
              
              add_answer_start    = True,

              stream_text   = False,
              request_id    = None,
              request_manager   = None,
              stream_callback   = None,
              wait_finalization = False,

              callbacks = None,
              predicted = None,
              
              _inference_manager    = None,
              
              ** kwargs
             ):
        """
            Performs inference (i.e., generate LLM answer based on the given text/messages).
            
            Arguments :
                - text  : the input query. If `messages` is provided, it can be `None`
                
                - prefix / format   : used to format `text` (only relevant if `text` is provided)
                
                - conv  : the `Conversation` history
                - messages  : the messages history
                - message_selector  : the message selection scheme (name or `MessageSelector`)
                - max_input_length  : maximum number of input tokens
                
                - tools : list of tools for the model
                - max_depth : maximum recursion depth for tool calls
                - allow_code_execution  : whether the model can execute python code or not
                
                - stop_words    : a list of words that stop the inference
                - max_new_tokens    : maximum number of tokens to generate
                - allowed_answers   : a list of possible answers that the model can generate
                
                - add_answer_start  : whether to add `answer_start` in the output
                                      `answer_start` is used to force the generation to
                                      start by a given string
                
                - stream_text   : whether to pass string (decoded text) or tokens to `stream_callback`
                - request_id    : used to identify the request for `request_manager`
                - request_manager   : `callable` that manages the request, see below for more info
                - stream_callback   : `callable` called at each inference step
                - wait_finalization : whether to wait request finalization or not (see below)
                
                - kwargs    : forwarded to `self.get_input` and `self.model`
            Return :
                - output    : `dict` containing predicted text + general information
            
            The `inference_manager` allows to control generation, such as aborting it before the end
            Here are the methods supported :
                - `build()` : initialize the request manager
                - `__call__(inference_item, request_id = None) -> bool` :
                    Call the manager at each step, passing tokens or decoded text (cf `stream_text`) :
                    If the returned value is `False`, the request is aborted
                - `wait_finalization(request_id) -> bool` :
                    only used if `wait_finalize` is True. Wait until the request is finalized.
                    If the returned value is `False`, the request is aborted.
                - `finalize(request_id)` :
                    finalize the request if not aborted (called at most once per request_id)
                - `pop(request_id)` :
                    Pop the request from the manager. Called when the request is aborted
        """
        ##############################
        #    State initialization    #
        ##############################
        
        _root_call = False
        if _inference_manager is None:
            _root_call = True
            
            if max_input_length is None: max_input_length = float('inf')
            if self.max_input_length:
                max_input_length = min(max_input_length, self.max_input_length)
            
            if conv is None:
                if messages is None:
                    conv = self.get_conv(conv_id = conv_id, ** kwargs)
                else:
                    conv = Conversation(messages = [Message(** msg) for msg in messages])
        
            if tools:
                tools = normalize_tools(tools)
            else:
                tools = []
            
            tool_names = ['print'] + [tool.name for tool in tools]
            if self.runtime == 'trt_llm':
                if tools or allow_code_execution:
                    kwargs['stop_condition'] = partial(_contains_code, tool_names = tool_names)
        
                if possible_answers:
                    kwargs['allowed_tokens'] = pad_batch(
                        self.encode_text(
                            possible_answers, add_sos_and_eos = False, return_type = 'np'
                        ),
                        pad_value = self.blank_token_idx,
                        dtype = 'int32'
                    )
            elif tools or allow_code_execution or possible_answers:
                raise NotImplementedError('The arguments `tools`, `allow_code_execution` and `possible_answers` are only supported with `TensorRT-LLM` runtime')

            _inference_manager  = InferenceManager(
                initial_state   = conv.get_state(),
                
                tokenizer   = self.tokenizer,
                stream_text = stream_text,
                
                callback    = stream_callback,
                request_id  = request_id,
                request_manager = request_manager,
                wait_finalization   = wait_finalization
            )
            
            kwargs.update(_inference_manager.get_inference_config())
        else:
            tool_names = ['print'] + [tool.name for tool in tools]
        
        ####################
        #   Prepare input  #
        ####################
        
        query = text
        if format:
            if prefix: prefix = format_text(prefix, ** kwargs)
            text = format_text(format, text = text, prefix = prefix, ** kwargs)

        messages    = get_messages(
            conv,
            message_selector,
            max_length  = max_input_length,
            tokenizer   = self.tokenizer,
            ** kwargs
        )
        
        prompt, tokens = self.get_input(
            text,
            messages    = messages,
            instructions    = conv.instructions,
            pinned_messages = conv.pinned,
            python_tools    = tools,
            allow_code_execution    = allow_code_execution,

            max_length  = max_input_length,
            return_text = True,
            
            ** kwargs
        )
        
        if _inference_manager.is_aborted():
            return {}
        
        ####################
        #     Inference    #
        ####################
        
        out = self.compiled_infer(
            tokens[None], max_new_tokens = max_new_tokens, tokenizer = self.tokenizer, ** kwargs
        )
        if self.runtime == 'trt_llm':
            _inference_manager.set_inference_stream(out)
            out = _inference_manager.result()
        
        if _inference_manager.is_aborted():
            return {}

        pred = self.decode_output(out)[0]
        if isinstance(pred, list) and len(pred) == 1: pred = pred[0]
        if add_answer_start and kwargs.get('answer_start', None):
            pred = kwargs['answer_start'] + pred

        code_block = None
        if tools or allow_code_execution:
            if 'print' in pred:
                pred = remove_simulated_output(pred)

            _, code_block = extract_code(pred)
            if code_block: code_block = code_block[0]

        _inference_manager.append(pred)
        
        if text:
            conv.append(text, role = 'user', query = query, format = format, ** kwargs)
        conv.append(pred, role = 'assistant', prompt = prompt)

        if code_block and _contains_tool(code_block, tool_names):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Tool call detected :\n{}'.format(code_block))

            code_result  = execute_code(
                code_block, tools = tools, conv = conv, model = self, ** kwargs
            )
            tool_message = format_code_result(code_result)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Code execution result : {}'.format(tool_message))

            if tool_message:
                conv.append(
                    tool_message, role = 'code', code = code_block, result = code_result
                )

                if len(_inference_manager) >= max_depth:
                    tools   = []
                    allow_code_execution    = False
                
                return self.infer(
                    None,
                    conv    = conv,
                    message_selector    = message_selector,
                    max_input_length    = max_input_length,

                    tools   = tools,
                    max_depth   = max_depth,
                    allow_code_execution    = allow_code_execution,

                    max_new_tokens  = max_new_tokens,
                    add_answer_start    = add_answer_start,

                    _inference_manager  = _inference_manager,
                    
                    _add_prompts    = False,

                    ** kwargs
                )

        full_output = _inference_manager.cumulated_results()
        full_output = full_output[0] if len(full_output) == 1 else '\n\n'.join(full_output)
        result = {
            'predicted' : full_output,
            
            'query' : query,
            'format'    : format,
            'prompt'    : prompt,
            'input_tokens'  : tokens,
            ** kwargs
        }

        if not _root_call:
            return result
        elif not _inference_manager.finalize():
            return result
        elif callbacks:
            apply_callbacks(callbacks, {}, result, save = False)
        
        return result

    answer  = add_prompt_wrapper('answer', fn = infer)
    
    ask_expert  = add_prompt_wrapper('expert',      fn = answer)
    translate   = add_prompt_wrapper('translate',   fn = answer)
    reformulate = add_prompt_wrapper('reformulate', fn = answer)
    describe    = add_prompt_wrapper('describe',    fn = answer)
    summarize   = add_prompt_wrapper('summarize',   fn = answer)
    extract_entities    = add_prompt_wrapper('extract_entities', fn = answer)
    
    @add_prompt_wrapper('rag')
    def rag(self,
            question,
            queries = None,
            *,
            
            k   = 10,
            reverse = True,
            
            paragraphs  = None,
            documents   = None,
            web_search  = None,
            search_config   = {},
            
            retriever   = None,
            retriever_config    = {},
            
            ** kwargs
           ):
        assert paragraphs or documents or web_search
        
        if queries is None:             queries = [question]
        elif isinstance(queries, str):  queries = [queries]
        
        if paragraphs is None:                  paragraphs = []
        elif hasattr(paragraphs, 'to_dict'):    paragraphs = paragraphs.to_dict('records')
        else:                                   paragraphs = paragraphs.copy()
        
        if web_search:
            for para in search_on_web(question, ** kwargs)['results'].values():
                paragraphs.extend(para)
        
        if isinstance(retriever, str):
            from models import get_pretrained
            retriever = get_pretrained(retriever)
        
        database    = retriever.predict(
            paragraphs, documents = documents, ** {** kwargs, ** retriever_config}
        )
        retrieved = retriever.retrieve(queries, database, k = k, reverse = reverse)

        if len(retrieved) > 1:
            infos = []
            for res in retrieved: infos.extend(res)
            infos = sorted(infos, key = lambda p: p['score'], reverse = not reverse)
        else:
            infos = retrieved[0]
        
        return self.answer(question, paragraphs = infos, ** kwargs)

def _contains_tool(text, tool_names):
    return any(name + '(' in text for name in tool_names)

def _contains_code(text, tool_names = []):
    if text.rstrip().endswith('```') and '```python' in text:
        return _contains_tool(text, tool_names)
    else:
        return False
