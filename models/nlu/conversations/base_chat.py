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
import uuid

from typing import List, Dict, Any
from dataclasses import dataclass, field

from utils import load_json, dump_json

@dataclass(order = True)
class Message:
    """
        A `Message` is a simple unit in a conversation containing any kind of data (text, image, ...)
        
        Arguments :
            - content   : the message content (text, filename, ...)
            - lang      : the lang of the message (en, fr, ...)
            - content_type  : the type of message ('text', 'image', 'audio', ...)
            
            - role  : the user role in the conversation ('user', 'assistant', 'tool', ...)
            - user  : the user name that sent the message (None if `type != 'user'`)
            
            - id    : the message unique id
            - conv_id   : the conversation id
            
            - time  : the time at which the message was sent
            - infos : additional information
    """
    content : str   = field(compare = False)
    lang    : str   = field(default = 'en', compare = False, repr = False)
    content_type    : str   = field(default = 'text', compare = False, repr = False)
    
    role    : str = field(default = 'assistant', compare = False)
    user    : str = field(default = None, compare = False)
    
    id      : Any = field(default_factory = uuid.uuid4, compare = False, repr = False)
    conv_id : Any = field(default = None, compare = False, repr = False)
    
    time    : float = field(default_factory = time.time, repr = False)
    infos   : Dict[str, Any]    = field(default_factory = dict, compare = False, repr = False)
    
    @property
    def text(self):
        return self.content

    def __getitem__(self, key):
        if key == 'text': return self.text
        return self.__dict__[key] if hasattr(self, key) else self.infos[key]
    
    def __setitem__(self, key, value):
        if key == 'text': self.content = value
        elif key in self.__dataclass_fields__: setattr(self, key, value)
        else: self.infos[key] = value
    
    def __contains__(self, key):
        return key == 'text' or key in self.__dict__ or key in self.infos
    
    def filter(self, *, all_match = True, ** kwargs):
        fn = all if all_match else any
        return fn(getattr(self, k) == v for k, v in kwargs.items())

    def get_config(self):
        config = self.__dict__.copy()
        config.update(config.pop('infos'))
        return config
    
@dataclass(unsafe_hash = True)
class Conversation:
    """
        A `Conversation` isa list of messages belonging to a given topic / conversation.
        This is similar to openai/anthropic "new chat" feature, a "chat" being equivalent to a `Conversation`. 
        
        Arguments :
            - id    : the conversation id
            - name  : the conversation name (for readability)
            - messages  : the list of messages in this conversation
            
            - metadata  : additional informations
            - system_prompt : system prompt used in this conversation
        
        Note that a conversation can be single- or multi- users
    """
    id      : Any   = field(default_factory = uuid.uuid4)
    name    : str   = field(default = None, hash = False)
    messages    : List[Message] = field(default_factory = list, hash = False, repr = False)
    pinned      : List[Message] = field(default_factory = list, hash = False, repr = False)
    instructions    : List[Message] = field(default_factory = list, hash = False, repr = False)
    
    metadata    : Dict[str, Any] = field(default_factory = dict, hash = False, repr = False)
    system_prompt   : str = field(default = None, hash = False, repr = False)
    
    __state_fields__    = ('messages', 'pinned', 'instructions')
    
    @property
    def users(self):
        return set([msg.user for msg in self.messages if msg.role == 'user'])
    
    @property
    def is_multi_users(self):
        return len(self.users) > 1
    
    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, idx):
        return self.messages[idx]
    
    def __contains__(self, message_id):
        return any(msg.id == message_id for msg in reversed(self.messages))
    
    def get_state(self):
        return {k : self.__dict__[k].copy() for k in self.__state_fields__}
    
    def set_state(self, state):
        self.__dict__.update(state)
    
    def index(self, message_id):
        for idx in reversed(range(len(self.messages))):
            if self.messages[idx].id == message_id:
                return idx
        raise IndexError('The message id {} is not in this conversation'.format(message_id))
    
    def append(self, content, *, pinned = False, ** kwargs):
        if 'role' not in kwargs and 'user' in kwargs: kwargs['role'] = 'user'
        infos = {
            k : kwargs.pop(k) for k in list(kwargs.keys())
            if k not in Message.__dataclass_fields__
        }
        message = Message(content = content, conv_id = self.id, infos = infos, ** kwargs)
        
        if not pinned:
            self.messages.append(message)
        else:
            self.pinned.append(message)
    
    def delete(self, *, from_id = None, last_n = 1):
        if from_id:
            if from_id not in self: raise ValueError('`{}` is not a valid message id'.format(from_id))
            self.messages = self.messages[self.index(from_id) - 1 :]
        else:
            n = 0
            for idx in reversed(range(len(self))):
                if self[idx].role == 'user':
                    n += 1
                    if n == last_n:
                        self.messages = self.messages[idx - 1 :]
                        return
    
    def filter(self, *, all_match = True, ** kwargs):
        if not self.messages: return []
        kwargs = {k : v for k, v in kwargs.items() if hasattr(self.messages[0], k)}
        return [msg for msg in self.messages if msg.filter(all_match = all_match, ** kwargs)]
    
    def save(self, filename):
        return dump_json(filename, self, indent = 2)
    
    @classmethod
    def load(cls, filename):
        conv = load_json(filename, default = {}) if isinstance(filename, str) else filename
        conv['messages'] = [Message(** msg) for msg in conv['messages']]
        
        return cls(** conv)

@dataclass(unsafe_hash = True)
class Chat:
    """
        A `Chat` is group of conversations associated to a given (set of) user(s). This can be seen as a channel in discord application, or like a user-specific interface in openai/anthropic. 
        
        Arguments :
            - id    : unique chat identifier (typically the channel id for discord, or user id for user-specific applications)
            - name  : a comprehensive name for this chat
            - platform  : the platform name (may be included in the prompt ?)
            
            - last_conv_id  : internally handled, used to identify the last conversation
            - conversations : the list of conversations within this chat.
                              In user-specific applications (e.g., openai/anthropic, this is equivalent to the list of "chat"s).
                              In messaging applications, like discord, it is more like "threads". However, in general channels, it may be only 1 conversation per chat, containing all the messages from this channel.
        
        Note : in order to not load the entire chat by default, the `save` and `load` methods require a directory, in order to save each conversation in a separate `json` file.
    """
    id  : Any   = field(default = 'default')
    name    : str   = field(default = None, hash = False)
    platform    : str   = field(default = None, hash = False)
    
    last_conv_id    : Any   = field(default = None, hash = False)
    conversations   : List[Conversation] = field(default_factory = list, hash = False, repr = False)
    
    directory   : str   = field(default = None, repr = False)
    unloaded_conv   : List[str] = field(default_factory = list, repr = False)
    
    @property
    def last_conv(self):
        return self[self.last_conv_id] if self.last_conv_id else None
    
    def _maybe_load_conv(self, conv_id):
        if conv_id not in self.unloaded_conv: return
        self.unloaded_conv.remove(conv_id)
        self.conversations.append(Conversation.load(os.path.join(
            self.directory, '{}.json'.format(conv_id)
        )))
    
    def __contains__(self, conv_id):
        return any(conv.id == conv_id for conv in self.conversations) or conv_id in self.unloaded_conv
    
    def __getitem__(self, conv_id):
        self._maybe_load_conv(conv_id)
        
        for conv in reversed(self.conversations):
            if conv.id == conv_id: return conv
        raise IndexError('Id {} is invalid'.format(conv_id))
    
    def get_conv(self, conv_id = None, message_id = None, create = True, ** _):
        if conv_id:
            if conv_id in self:
                return self[conv_id]
            elif create:
                conv = Conversation(id = conv_id)
                self.conversations.append(conv)
                self.last_conv_id = conv.id
                return conv
            else:
                return None
        elif message_id:
            return self.get_conv_from_message_id(message_id)
        elif create and self.last_conv_id is None:
            conv = Conversation()
            self.conversations.append(conv)
            self.last_conv_id = conv.id
            return conv
        else:
            return self.last_conv
    
    def get_conv_from_message_id(self, message_id):
        for conv in reversed(self.conversations):
            if any(msg.id == message_id for msg in reversed(conv.messages)):
                return conv
        raise ValueError('The message {} is not in this chat'.format(message_id))
    
    def append(self, content, *, conv = None, conv_id = None, new_conv = False, ** kwargs):
        if conv is not None:
            if conv.id not in self: self.conversations.append(conv)
        elif conv_id:
            conv = self[conv_id]
        elif not new_conv and self.conversations:
            conv = self.last_conv
        else:
            conv = Conversation(name = conv_name)
            self.conversations.append(conv)
        
        conv.append(content, ** kwargs)
        self.last_conv_id = conv.id
    
    def save(self, directory):
        os.makedirs(directory, exist_ok = True)
        for conv in self.conversations:
            conv.save(os.path.join(directory, '{}.json'.format(conv.id)))
        
        filename = os.path.join(directory, 'config.json')
        return dump_json(
            filename, {k : v for k, v in self.__dict__.items() if k != 'conversations'}, indent = 4
        )
    
    @classmethod
    def load(cls, directory):
        config = load_json(os.path.join(directory, 'config.json'), default = {})
        config['directory']     = directory
        config['unloaded_conv'] = [
            f[:-5] for f in os.listdir(conversations) if f != 'config.json'
        ]
        return cls(** config)
