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

from utils.text import parse_document

def select_all(query, items, *, tokenizer, max_items = None, max_length = None, directory = None, ** kwargs):
    if 'documents' in kwargs:
        items = parse_document(
            kwargs['documents'],
            cache_dir       = os.path.join(directory, '.cache') if directory else None,
            image_folder    = os.path.join(directory, '.cache', '{}') if directory else None
        )
    
    if not max_length:
        return items[:max_items], None
    
    selected, total_length = [], 0
    for item in items[:max_items]:
        if not item.get('length', None) and isinstance(item.get('content', None), str):
            item['length'] = len(tokenizer.tokenize(item['content']))
        
        selected.append(item)
        total_length += item.get('length', 0)
    
    return selected, total_length

select_first = select_all

def select_last(query, items, ** kwargs):
    items, length = select_all(query, items[::-1], ** kwargs)
    return items[::-1], length
