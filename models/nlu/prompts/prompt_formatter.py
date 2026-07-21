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

_default_prompt_order = (
    'system',
    'paragraphs',
    'python_tools',
    'personnality',
    'instructions'
)

class PromptFormatter:
    def __init__(self, tokenizer, audio_token = None, image_token = None, video_token = None):
        self.tokenizer = tokenizer

        self.audio_token = audio_token
        self.image_token = image_token
        self.video_token = video_token

        self._default_kwargs    = self.tokenizer.tokens.copy()
        self._default_kwargs.update({
            'audio_token'   : self.audio_token,
            'image_token'   : self.image_token,
            'video_token'   : self.video_token,

            'timestamp_to_str'  : timestamp_to_str
        })
        if 'strftime_now' in self.template:
            self._default_kwargs['strftime_now'] = lambda fmt: datetime.now().strftime(fmt)


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

    @property
    def default_kwargs(self):
        return self._default_kwargs

    def prepare_query(self, text, *, format = None, prefix = None, suffix = None, ** kwargs):
        if format: text = format_text(format, text = text, ** kwargs)
        if prefix: text = format_text(prefix, ** kwargs) + text
        if suffix: text = text + format_text(suffix, ** kwargs)

        return text

    @timer
    def get_prompt(self,
                   messages = None,
                   *,

                   prompt_order = _default_prompt_order,
                   prompt_format    = None,
                   history_prefix   = None,
                   message_format   = None,
                   last_message_format    = None,

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

            The heavy lifting is delegated to the pure module-level helpers (`normalize_messages`,
            `extract_formats`, `build_system_sections`, `normalize_message_content`,
            `apply_message_formats`, `assemble_system_prompt`) so each step is unit-testable on its
            own ; this method only orchestrates them and renders the final chat template.
        """
        with Timer('initialization'):
            messages, system = normalize_messages(messages, ** kwargs)
            if system is not None:
                if 'system' in kwargs:
                    raise RuntimeError(
                        '`system` kwargs and `role = "system"` cannot be provided together'
                    )
                kwargs['system'] = system

            formats = extract_formats(kwargs)

            kwargs.update(self.default_kwargs)
            kwargs['prompt_format'] = prompt_format
            if 'date_string' not in kwargs and 'date_string' in self.template:
                kwargs['date_string'] = timestamp_to_str(time.time(), include_time = False)

            prompt_data, multimodal_data = build_system_sections(
                prompt_order, formats, self.supported_types, kwargs
            )

        with Timer('messages preparation'):
            messages, message_multimodal = normalize_message_content(
                messages, self.supported_types, self.is_text_only
            )
            for key, items in message_multimodal.items():
                multimodal_data.setdefault(key, []).extend(items)

        with Timer('messages formatting'):
            messages = apply_message_formats(
                messages, message_format, last_message_format, kwargs
            )

            system_prompt = assemble_system_prompt(
                prompt_data, history_prefix, messages, kwargs
            )
            messages.insert(0, {'role' : 'system', 'content' : system_prompt})

        prompt = format_text(
            self.template,
            messages = messages,
            add_generation_prompt = add_generation_prompt,
            ** kwargs
        )
        if add_generation_prompt and answer_start:
            prompt = prompt + answer_start

        return prompt, multimodal_data


def normalize_messages(messages, ** kwargs):
    """
        Normalize the `messages` argument into a `list[dict]`.

        Supports `None` (→ empty list), a single `dict`, a raw `str` (→ new user message),
        or a `list` of `dict` / `models.nlu.Message` (converted through `to_dict`).

        Return :
            - messages  : the normalized `list[dict]` (a fresh list, never the caller's)
            - system    : the `content` of a leading `role == 'system'` message (popped out),
                          or `None` if the first message is not a system message
    """
    if messages is None:
        return [], None
    if isinstance(messages, dict):
        return [messages], None
    if isinstance(messages, str):
        return [_build_message(messages, ** kwargs)], None
    if not isinstance(messages, list):
        raise ValueError('Unsupported `messages` type ({}) : {}'.format(
            type(messages), messages
        ))

    messages = messages.copy()
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            messages[i] = message.to_dict()

    system = None
    if messages and messages[0]['role'] == 'system':
        system = messages.pop(0)['content']

    return messages, system


def extract_formats(kwargs):
    """
        Pop every `*_format` entry out of `kwargs` and return them as a `{name : format}`
        mapping (the `_format` suffix is stripped from the key). `kwargs` is mutated in place.
    """
    return {
        key[:-7] : kwargs.pop(key) for key in list(kwargs.keys()) if key.endswith('_format')
    }


def build_system_sections(prompt_order, formats, supported_types, kwargs):
    """
        Build the ordered system-prompt sections from `prompt_order`.

        For each key of `prompt_order` present (and truthy) in `kwargs`, the section is either
        rendered through its matching entry in `formats` or appended as-is. Multimodal parts
        found in list-valued sections (e.g. `paragraphs`) are collected along the way.

        Return :
            - prompt_data       : the `list` of rendered section strings, in order
            - multimodal_data   : a `{type : [parts]}` mapping of the collected multimodal parts
    """
    multimodal_types = supported_types[1:]

    prompt_data, multimodal_data = [], {}
    for key in prompt_order:
        value = kwargs.get(key, None)
        if not value:
            continue

        if key in formats:
            prompt_data.append(format_text(formats[key], ** kwargs))
        else:
            prompt_data.append(value)

        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                for part in item.get('content', [item]):
                    if isinstance(part, dict) and part.get('type', 'text') in multimodal_types:
                        multimodal_data.setdefault(part['type'], []).append(part)

    return prompt_data, multimodal_data


def normalize_message_content(messages, supported_types, is_text_only):
    """
        Filter each message `content` against `supported_types`.

        When `is_text_only`, multimodal parts are dropped and the remaining texts joined into a
        single string. Otherwise unsupported parts are filtered out, a lone text part is collapsed
        to a plain string, and the surviving non-text parts are collected as multimodal data.

        `messages` is mutated in place (matching the previous behaviour) and returned.

        Return :
            - messages          : the (possibly mutated) message list
            - multimodal_data   : a `{type : [parts]}` mapping of the collected multimodal parts
    """
    multimodal_data = {}
    if not any(isinstance(msg['content'], list) for msg in messages):
        return messages, multimodal_data

    for i, message in enumerate(messages):
        if isinstance(message['content'], str):
            continue

        message = message.copy()
        if is_text_only:
            message['content'] = '\n\n'.join([
                c['text'] for c in message['content'] if c.get('text', None)
            ])
        else:
            message['content'] = [
                c for c in message['content'] if c['type'] in supported_types
            ]

            if len(message['content']) == 1 and message['content'][0]['type'] == 'text':
                message['content'] = message['content'][0]['text']
            else:
                for c in message['content']:
                    if c['type'] != 'text':
                        multimodal_data.setdefault(c['type'], []).append(c)

        messages[i] = message

    return messages, multimodal_data


def apply_message_formats(messages, message_format, last_message_format, kwargs):
    """
        Apply `message_format` to every non-leading-system message and `last_message_format`
        to the last message. `messages` is mutated in place and returned.
    """
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

    return messages


def assemble_system_prompt(prompt_data, history_prefix, messages, kwargs):
    """
        Join the system-prompt `prompt_data` sections with blank lines, optionally appending the
        rendered `history_prefix` when the conversation holds more than one user message.
    """
    prompt_data = list(prompt_data)
    if history_prefix and sum(msg['role'] == 'user' for msg in messages) > 1:
        prompt_data.append(format_text(history_prefix, messages = messages, ** kwargs))

    return '\n\n'.join(prompt_data)


def _build_message(content, user = None, ** kwargs):
    return {
        'role'      : 'user',
        'content'   : content,
        'time'      : time.time(),
        'user'      : user
    }
