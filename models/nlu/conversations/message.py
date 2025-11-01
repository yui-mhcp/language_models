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

from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class Message:
    """
        A `Message` is a simple unit in a conversation.
        It can contain any kind of data (text, image, ...)
        
        Arguments :
            - role      : the role of the message sender ('user', 'assistant', 'system', ...)
            - content   : the message content (text, filename, ...)
            - content_type  : the type of content ('text', 'image', 'audio', ...)
            
            - user  : the user name that sent the message (None if `type != 'user'`)
            
            - id    : the message unique id
            - conv_id   : the conversation id
            
            - time  : the time at which the message was sent
            - infos : additional information
    """
    role    : str
    content : str
    content_type    : str   = field(default = 'text', repr = False)
    
    user    : str = field(default = None)
    
    id      : Any = field(default_factory = uuid.uuid4, repr = False)
    conv_id : Any = field(default = None, repr = False)
    
    time    : float = field(default_factory = time.time, repr = False)
    metadata    : Dict[str, Any]    = field(default_factory = dict, repr = False)
    
    @property
    def text(self):
        return self.content

    def __getitem__(self, key):
        if key == 'text':           return self.text
        elif key in self.__dict__:  return self.__dict__[key]
        elif key in self.metadata:     return self.metadata[key]
        else: raise KeyError('`Message` has no attribute {}'.format(key))
    
    def __setitem__(self, key, value):
        if key == 'text': self.content = value
        elif key in self.__dataclass_fields__: setattr(self, key, value)
        else: self.metadata[key] = value
    
    def __contains__(self, key):
        return key == 'text' or key in self.__dict__ or key in self.metadata
    
    def get(self, key, * args):
        if args and key not in self: return args[0]
        return self[key]
        
    def filter(self, *, full_match = True, ** kwargs):
        fn = all if full_match else any
        return fn(hasattr(self, k) and getattr(self, k) == v for k, v in kwargs.items())
    
    def to_json(self):
        data = self.__dict__.copy()
        data.update(data.pop('metadata'))
        return data
    
    to_dict = to_json