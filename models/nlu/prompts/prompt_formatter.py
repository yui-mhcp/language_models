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

from datetime import datetime
from functools import cached_property

from loggers import Timer, timer
from utils import timestamp_to_str
from utils.text import format_text

format_text = timer(format_text)

class PromptFormatter:
    def __init__(self, tokenizer, audio_token = None, image_token = None, video_token = None):
        self.tokenizer = tokenizer
        
        self.audio_token = audio_token
        self.image_token = image_token
        self.video_token = video_token
    
    @property
    def template(self):
        return self.tokenizer.template
    
    @property
    def is_text_only(self):
        return not self.image_token and not self.video_token and not self.audio_token
    
    @property
    def is_omni(self):
        return self.image_token and self.video_token and self.audio_token

    @cached_property
    def supported_types(self):
        supported = ['text']
        if self.image_token: supported.append('image')
        if self.video_token: supported.append('video')
        if self.audio_token: supported.append('audio')
        return supported
    
    def prepare_query(self, text, *, format = None, prefix = None, suffix = None, ** kwargs):
        if format: text = format_text(format, text = text, ** kwargs)
        if prefix: text = format_text(prefix, ** kwargs) + text
        if suffix: text = text + format_text(suffix, ** kwargs)
        
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

        multimodal_data = {}
        if not self.is_text_only and 'paragraphs_format' in kwargs and kwargs.get('paragraphs', ()):
            for para in kwargs.get('paragraphs', []):
                content = para.get('content', [para])
                for c in content:
                    if c['type'] in self.supported_types[1:]:
                        multimodal_data.setdefault(c['type'], []).append(c)

        with Timer('initialization'):
            formats = {k[:-7] : kwargs.pop(k) for k in list(kwargs.keys()) if k.endswith('_format')}

            kwargs.update(self.tokenizer.tokens)
            kwargs.update({
                'audio_token'   : self.audio_token,
                'image_token'   : self.image_token,
                'video_token'   : self.video_token,
                
                'prompt_format' : prompt_format,
                'timestamp_to_str'  : timestamp_to_str
            })
            if 'date_string' not in kwargs and 'date_string' in self.template:
                kwargs['date_string'] = timestamp_to_str(time.time(), include_time = False)
            elif 'strftime_now' in self.template:
                kwargs['strftime_now'] = lambda fmt: datetime.now().strftime(fmt)

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
                for i, message in enumerate(messages):
                    if not isinstance(message, dict):
                        messages[i] = message.to_dict()

            if text:
                messages.append(_build_message(text, ** kwargs))
            
            if any(isinstance(msg['content'], list) for msg in messages):
                for i, message in enumerate(messages):
                    if isinstance(message['content'], str):
                        continue

                    message = message.copy()
                    if self.is_text_only:
                        message['content'] = '\n\n'.join([
                            c['text'] for c in message['content'] if c.get('text', None)
                        ])
                    else:
                        message['content'] = [
                            c for c in message['content'] if c['type'] in self.supported_types
                        ]

                        if len(message['content']) == 1 and message['content'][0]['type'] == 'text':
                            message['content'] = message['content'][0]['text']
                        else:
                            for c in message['content']:
                                if c['type'] != 'text':
                                    multimodal_data.setdefault(c['type'], []).append(c)

                    messages[i] = message
        
        with Timer('messages formatting'):
            if message_format:
                for i, message in enumerate(messages):
                    if i == 0 and message['role'] == 'system':
                        continue
                    
                    message = message.copy()
                    if isinstance(message['content'], str):
                        message['content'] = format_text(
                            message_format, text = message['content'], message = message, ** kwargs
                        )
                    else:
                        message['content'] = message['content'].copy()
                        message['content'].insert(0, {'type' : 'text', 'text' : format_text(
                            message_format, text = message['content'], message = message, ** kwargs
                        )})

                    messages[i] = message

            if last_message_format:
                last_msg  = messages[-1].copy()
                formatted = format_text(
                    last_message_format, text = last_msg['content'], message = last_msg, ** kwargs
                )
                if isinstance(last_msg['content'], str):
                    last_msg['content'] = formatted
                else:
                    last_msg['content'] = [{'type' : 'text', 'text' : formatted}] + last_msg['content']
                messages[-1] = last_msg

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
        
        if add_generation_prompt and answer_start:
            prompt = prompt + answer_start
        
        return prompt, multimodal_data

def _build_message(content, user = None, ** kwargs):
    return {
        'role'      : 'user',
        'content'   : content,
        'time'      : time.time(),
        'user'      : user
    }