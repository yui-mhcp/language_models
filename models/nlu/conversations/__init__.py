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

from .base_chat import Chat, Conversation, Message
from .message_selector import MessageSelector, LastMessageSelector

_selectors = {
    k.replace('MessageSelector', '').lower() : v for k, v in globals().items()
    if isinstance(v, type) and issubclass(v, MessageSelector)
}

def get_message_selector(name, ** kwargs):
    normalized = name.replace('MessageSelector', '').lower()
    if normalized not in _selectors:
        raise ValueError('Unknown message selector : {}'.format(name))
    
    return _selectors[normalized](** kwargs)