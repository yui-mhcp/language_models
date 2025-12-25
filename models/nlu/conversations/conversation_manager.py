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
import warnings

from loggers import Timer, timer
from .conversation import Conversation
from .conv_item_selectors import select_items

logger = logging.getLogger(__name__)

MIN_LENGTH_THRESHOLD = 10

_in_memory_conv_id = '__in_memory__'
_default_conv_id    = 'default'

class ConversationManager:
    def __init__(self, path, tokenizer):
        self.path   = path
        self.tokenizer = tokenizer
        
        self._convs = {}
    
    @timer
    def get_context(self,
                    conv,
                    query,
                    *,
                    
                    directory   = None,
                    documents   = None,
                    attachments = None,
                    use_specified_documents = False,
                    
                    messages_selector   = 'last',
                    paragraphs_selector = None,
                    instructions_selector   = 'all',
                    
                    max_length = None,
                    selection_order = ('instructions', 'messages', 'paragraphs'),
                    
                    ** kwargs
                   ):
        if directory is None: directory = self.path
        if directory:         directory = os.path.join(directory, conv.id)
        
        kwargs.update({
            'directory' : directory,
            'tokenizer' : self.tokenizer,
            'messages_selector' : messages_selector,
            'paragraphs_selector'   : paragraphs_selector,
            'instructions_selector' : instructions_selector
        })
        
        if attachments and 'Attachments' not in query:
            if isinstance(attachments, str): attachments = [attachments]
            query = 'Attachments:{}\n\n{}'.format(
                ' ' + attachments[0] if len(attachments) == 1 else ''.join([
                    '\n- ' + f for f in attachments
                ]), query
            )
        
        remaining_length = max_length
        default_selector = kwargs.pop('selector', None)
        
        context = {}
        for k in selection_order:
            if remaining_length and remaining_length <= MIN_LENGTH_THRESHOLD:
                warnings.warn('The context length is about to exceed `max_length`, stopping item selection')
                break

            item_selector = kwargs.get('{}_selector'.format(k), None) or default_selector
            if not item_selector:
                continue
            
            config = kwargs.copy()
            if '{}_selector_config'.format(k) in kwargs:
                config.update(config.pop('{}_selector_config'.format(k)))
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Using `{}` selector for key `{}`'.format(item_selector, k))
            
            with Timer('{}_selector'.format(k)):
                if k == 'paragraphs':
                    config.update({
                        'items' : conv.documents,
                        'documents' : documents if use_specified_documents else conv.documents
                    })
                else:
                    config['items'] = getattr(conv, k)
                
                context[k], length = select_items(
                    query,
                    conv    = conv,
                    selector    = item_selector,
                    max_length  = remaining_length,
                    ** config
                )
            if remaining_length: remaining_length -= length
        
        return context

    def get_conversation(self,
                         conv_id = None,
                         
                         *,
                         
                         directory  = None,
                         
                         messages   = None,
                         instructions   = None,
                         
                         ** kwargs
                        ):
        if directory is None: directory = self.path
        
        if conv_id is None:
            conv_id = _in_memory_conv_id
            if messages is not None: self._convs.pop((directory, conv_id), None)
        elif (directory, conv_id) not in self._convs:
            path = self.get_conv_file(directory, conv_id)
            if os.path.exists(path):
                self._convs[(directory, conv_id)] = Conversation.load(path)
        
        conv = self._convs.get((directory, conv_id), None)
        if conv is None:
            kwargs = {k : v for k, v in kwargs.items() if k in Conversation.__dataclass_fields__}
            kwargs['id'] = conv_id
            conv   = Conversation(** kwargs)
            
            if messages:
                for msg in messages: conv.add_message(msg)
            
            if instructions:
                if isinstance(instructions, str): instructions = [instructions]
                for inst in instructions: conv.add_instruction(inst)

            self._convs[(directory, conv.id)] = conv

        return conv

    def save(self, conv, directory = None):
        if conv.id != _in_memory_conv_id:
            conv.save(self.get_conv_file(directory or self.path, conv.id))
    
    @staticmethod
    def get_conv_file(directory, conv_id):
        return os.path.join(directory, conv_id, 'conversation.json')
