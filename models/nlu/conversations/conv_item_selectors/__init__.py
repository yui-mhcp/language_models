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

from functools import partial

from utils.generic_utils import import_submodules

_globals = {}
_selectors = {}
for module in import_submodules(__package__):
    for k, v in vars(module).items():
        if k.startswith('select_'):
            _globals[k] = v
            _selectors[k[7:].lower()] = v
        elif k.endswith('Selector'):
            _globals[k] = v
            _selectors[k[:-8].lower()] = v

globals().update(_globals)
del _globals

def get_selector(selector, ** kwargs):
    """ Return the item selector (callable or class) associated to the given `name` """
    if isinstance(selector, dict):
        kwargs.update(selector)
        selector = kwargs.pop('name')
    
    if isinstance(selector, str):
        selector = selector.lower()
        if selector.startswith('select_'):  selector = selector[7:]
        elif selector.endswith('selector'): selector = selector[:-8]
        
        selector = _selectors.get(selector, None)

    if isinstance(selector, type):
        return selector(** kwargs)
    elif callable(selector):
        return selector if not kwargs else partial(selector, ** kwargs)
    else:
        raise ValueError('Unsupported selector : {}'.format(name))

def select_items(query, items, *, selector = 'last', ** kwargs):
    """
        Call the item selection strategy (`selector`) on the given conversation (`conv`)
        
        Arguments :
            - query : the user formatted message (typically used in similarity-based search)
            - items : the list of candidates in which to select
            - selector  : the message selection strategy
                          - str : the name of the method/class to use
                          - callable    : a function returning a list of `dict` from `items`
                                          `selector(query, items, ** kwargs) -> List[dict]`
            - kwargs    : additional kwargs forwarded to the method
                    - conv  : the `Conversation` object
                    - tokenizer : the `Tokenizer` to use for item length computation
                    - directory : the path where to save files (if needed)
                    - documents : possible documents to use
        Return :
            - selected_items  : a `list` of `dict` from `items`
    """
    if not items: return [], 0

    if isinstance(selector, (str, dict, type)):
        selector = get_selector(selector, ** kwargs)
    
    return selector(query, items, ** kwargs)