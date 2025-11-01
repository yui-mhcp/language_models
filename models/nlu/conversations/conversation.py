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

import uuid

from typing import List, Dict, Any
from dataclasses import dataclass, field

from .message import Message
from utils import load_json, dump_json
from utils.text.parsers import parse_document
    
@dataclass(unsafe_hash = True)
class Conversation:
    """
        A `Conversation` is a list of `Message`.
        This is similar to openai/anthropic "new chat" feature, a "chat" being equivalent to a `Conversation`. 
        
        Arguments :
            - id    : the conversation id
            - name  : the conversation name (for readability)
            
            - prompts   : custom prompts for the conversation
                          should be a mapping `{prompt_key : prompt}`
            
            - messages  : the list of messages in this conversation
            - pinned    : messages pinned in the conversation
            - instructions  : special instructions for the model, specific to this conversation
            
            - documents : a list of documents (filenames) associated to this conversation
                          documents are stored as `Message` to keep track of metadata
            
            - metadata  : additional informations
        
        Note that a conversation can be single- or multi- users
        
        The `pinned` and `instructions` are forwarded to the tokenizer `encode_chat` method.
        They should therefore be used in one of the prompt format to be given to the model.
        By default, the `system_prompt` adds these special messages in the context.
    """
    id      : Any   = field(default_factory = uuid.uuid4, repr = False)
    name    : str   = field(default = None, hash = False)
    
    prompts : Dict[str, str]    = field(default_factory = dict, hash = False, repr = False)
    
    messages    : List[Message] = field(default_factory = list, hash = False, repr = False)
    pinned      : List[Message] = field(default_factory = list, hash = False, repr = False)
    instructions    : List[Message] = field(default_factory = list, hash = False, repr = False)
    
    documents   : List[Message]  = field(default_factory = list, hash = False, repr = False)
    metadata    : Dict[str, Any] = field(default_factory = dict, hash = False, repr = False)
    
    __state_fields__    = ('messages', 'pinned', 'instructions', 'documents')
    
    @property
    def users(self):
        return set(msg.user for msg in self.messages if msg.user is not None)
    
    @property
    def is_multi_users(self):
        return len(self.users) > 1
    
    @property
    def last_updated(self):
        return -1 if not self.messages else self.messages[-1].time
    
    @property
    def has_documents(self):
        return bool(self.documents)
    
    @property
    def paragraphs(self):
        paragraphs = []
        for doc in self.documents:
            paragraphs.extend(parse_document(doc.content))
        return paragraphs
    
    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, idx):
        return self.messages[idx]
    
    def __contains__(self, message_id):
        """ Returns whether `message_id` is in the conversation """
        return any(msg.id == message_id for msg in reversed(self.messages))
    
    def _append(self, _state, content, /, *, role = None, ** kwargs):
        if role is None: role = 'user' if 'user' in kwargs else 'assistant'
        
        metadata = {
            k : kwargs.pop(k) for k in list(kwargs.keys())
            if k not in Message.__dataclass_fields__
        }
        message = Message(
            content = content, role = role, conv_id = self.id, metadata = metadata, ** kwargs
        )
        _state.append(message)
        return message

    def add_document(self, document, *, content_type = 'document', ** kwargs):
        """ Add a new document to the conversation """
        if any(doc.content == document for doc in self.documents): return None
        return self._append(self.documents, document, content_type = content_type, ** kwargs)
    
    def add_instruction(self, instruction, *, content_type = 'text', ** kwargs):
        """ Add a new instruction to the conversation """
        if any(inst.content == instruction for inst in self.instructions): return None
        return self._append(self.instructions, instruction, content_type = content_type, ** kwargs)

    def add_message(self, text, *, content_type = 'text', ** kwargs):
        """ Add a new message to the conversation """
        if isinstance(text, dict):
            content = text.pop('content' if 'content' in text else 'text')
            if 'content_type' not in text: kwargs['content_type'] = content_type
            kwargs.update(text)
            return self._append(self.messages, content, ** kwargs)
        else:
            return self._append(self.messages, text, content_type = content_type, ** kwargs)

    append = add_message
    
    def get_state(self):
        return {k : self.__dict__[k].copy() for k in self.__state_fields__}
    
    def remove_document(self, document):
        if isinstance(document, int):
            return self.documents.pop(document)
        
        for idx in range(len(self.documents)):
            if self.documents[idx].content == document:
                return self.documents.pop(idx)
        raise IndexError('The filename `{}` is not in the conversation'.format(filename))
    
    def remove_instruction(self, instruction):
        if isinstance(instruction, int):
            return self.instructions.pop(instruction)
        
        for idx in range(len(self.instructions)):
            if self.instructions[idx].content == instruction:
                return self.instructions.pop(idx)
        raise IndexError('The instruction `{}` is not in the conversation'.format(instruction))
    
    def save(self, filename):
        return dump_json(filename, self, indent = 2)
    
    def set_state(self, state):
        self.__dict__.update(state)

    @classmethod
    def load(cls, filename):
        conv = load_json(filename, default = {}) if isinstance(filename, str) else filename
        for k in conv.__state_fields__:
            conv[k] = [Message(** msg) for msg in conv[k]]
        
        return cls(** conv)

def set_paragraph_content(paragraphs):
    if 'content_type' not in paragraphs[0]:
        paragraphs = [p.copy() for p in paragraphs]
        for p in paragraphs:
            p['content_type'] = p.pop('type', 'text')
            for k in ('text', 'filename', 'items', 'rows'):
                if k in p:
                    p['content'] = p.pop(k)
                    break
            else:
                logger.warning('Unable to identify the content for paragraph : {}'.format(p))
                p['content'] = None
    
    return [p for p in paragraphs if p['content']]
