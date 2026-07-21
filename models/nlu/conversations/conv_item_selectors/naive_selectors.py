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

from .utils import estimate_token_budget, parse_documents

def select_all(query, items, *, tokenizer, max_items = None, max_length = None, ** kwargs):
    if 'documents' in kwargs: items = parse_documents(** kwargs)
    
    if not max_length:
        return items[:max_items], None
    
    selected, total_length = [], 0
    for item in items[:max_items]:
        selected.append(item)
        total_length += estimate_token_budget(item, tokenizer)
        
        if total_length >= max_length:
            break
    
    return selected, total_length

select_first = partial(select_all, max_items = 10)

def select_last(query, items, *, documents = None, ** kwargs):
    if documents: items = parse_documents(documents, ** kwargs)
    items, length = select_first(query, items[::-1], ** kwargs)
    return items[::-1], length