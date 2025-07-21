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

from abc import ABC, abstractmethod

class MessageSelector(ABC):
    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)
    
    def __call__(self, conv, ** kwargs):
        return self.get_messages(conv, ** kwargs)

    @abstractmethod
    def get_messages(self, conv, *, chat = None, user = None, ** kwargs):
        """ Return a `list` of `Mesage` from the given `conv` """

class LastMessageSelector(MessageSelector):
    def get_messages(self, conv, *, tokenizer, max_length = None, max_messages = 5, ** kwargs):
        """ Return a `list` of `Mesage` from the given `conv` """
        if not max_length: max_length = float('inf')
        
        n, total_length = 0, 0
        messages = []
        for message in reversed(conv.messages):
            if message.role == 'user': n += 1
            
            if 'length' not in message or not message['length']:
                message['length'] = len(tokenizer.tokenize(message['content']))
            
            if (n) and (total_length + message['length'] > max_length or n > max_messages):
                break
            
            total_length += message['length']
            messages.append(message)
            
            if n == max_messages:
                break
        
        return messages[::-1]
            
