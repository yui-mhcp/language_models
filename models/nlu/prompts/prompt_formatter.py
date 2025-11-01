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

import time

from loggers import Timer, timer
from utils import timestamp_to_str
from utils.text import format_text

format_text = timer(format_text)

class PromptFormatter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @property
    def template(self):
        return self.tokenizer.template
    
    def prepare_query(self, text, *, format = None, prefix = None, ** kwargs):
        if format: text = format_text(format, text = text, ** kwargs)
        if prefix: text = format_text(prefix, ** kwargs) + text
        
        return text

    @timer
    def get_prompt(self,
                   text   = None,
                   *,

                   messages   = None,
                     
                   system_prompt  = None,
                   message_format = None,
                   last_message_format    = None,
                   
                   prompt_format  = None,
                   answer_start   = None,
                   add_generation_prompt  = True,
                     
                   format = None,
                   prefix = None,

                   ** kwargs
                  ):
        """
            Encode `messages` according to `self.chat_template`. This is used for chat-based LLM.
            
            Arguments :
                - text  : the new message (handled as the new user message)
                
                - messages  : `list` of messages (`dict` or `models.nlu.Message`)
                              should have at least `content` and `role` entries
                - paragraphs    : `list` of paragraphs (`dict`) with a `type` entry (text, table, ...)
                - answer_start  : string added after the chat template, used to guide the start of generation
                - add_generation_prompt : bool, whether to add generation special tokens or not
                                          This special argument is used in most of modern templates
                
                
                - system_prompt : custom system message (used as the first message)
                                  if `messages[0]['role'] == 'system'`, this argument is ignored
                - prompt_format : special string used in custom system prompts (`models.nlu.prompts`)
                - message_format    : special format applied to each message
                - paragraphs_format : custom string to format paragraphs
                                      arguments are `paragraphs` and `kwargs`
                
                - last_message_format   : special format applied only on the last message
            Return :
                - tokens    : the encoded chat message
            
            **Important Note**: the `paragraphs` and `prompt_format` are custom arguments used in the custom system prompts defined in `models.nlu.prompts`. If these arguments are not exploited in any of the format (e.g., system_prompt, message_format, last_message_format), they will be ignored.
            If `paragraphs_format` is provided, the `paragraphs` argument forwarded to the chat template will be a `string`, while if `paragraphs_format` is not provided, `paragraphs` will be forwarded directly no matter what it is.
        """
        ####################
        # Prepare formats  #
        ####################

        with Timer('initialization'):
            formats = {k[:-7] : kwargs.pop(k) for k in list(kwargs.keys()) if k.endswith('_format')}

            kwargs.update(self.tokenizer.tokens)
            kwargs.update({
                'prompt_format' : prompt_format,
                'timestamp_to_str'  : timestamp_to_str
            })
            if 'date_string' not in kwargs and 'date_string' in self.template:
                kwargs['date_string'] = timestamp_to_str(time.time(), include_time = False)

            for key, _format in formats.items():
                if kwargs.get(key, None):
                    kwargs[key] = format_text(_format, ** kwargs)

        ##############################
        #    Messages preparation    #
        ##############################
        
        with Timer('messages preparation'):
            if messages is None:
                messages = []
            elif isinstance(messages, dict):
                messages = [messages]
            elif isinstance(messages, str):
                messages = [_build_message(message, ** kwargs)]
            elif not isinstance(messages, list):
                raise ValueError('Unsupported `messages` type ({}) : {}'.format(
                    type(messages), messages
                ))
            else:
                messages = messages.copy()

            if text:
                messages.append(_build_message(text, ** kwargs))

            if message_format:
                for i, message in enumerate(messages):
                    if isinstance(message, dict):
                        message = message.copy()
                    else:
                        message = message.to_dict()

                    if i or message['role'] != 'system':
                        message['content'] = format_text(
                            message_format, text = message['content'], message = message, ** kwargs
                        )

                    messages[i] = message

            if last_message_format:
                if not message_format:
                    if isinstance(messages[-1], dict):
                        messages[-1] = messages[-1].copy()
                    else:
                        messages[-1] = messages[-1].to_json()

                messages[-1]['content'] = format_text(
                    last_message_format, text = messages[-1]['content'], message = messages[-1], ** kwargs
                )

            if system_prompt and messages[0]['role'] != 'system':
                messages.insert(0, {
                    'role'  : 'system',
                    'content'   : format_text(system_prompt, messages = messages, ** kwargs)
                })

        prompt = format_text(
            self.template,
            messages = messages,
            add_generation_prompt = add_generation_prompt,
            ** kwargs
        )
        
        if answer_start:
            prompt = prompt + answer_start
        
        return prompt, {}

def _build_message(content, content_type = 'text', user = None, ** kwargs):
    return {
        'role'      : 'user',
        'content'   : content,
        'content_type'  : content_type,
        'time'      : time.time(),
        'user'      : user
    }