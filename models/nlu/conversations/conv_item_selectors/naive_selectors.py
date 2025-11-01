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

_document_content_types = ('document', 'image', 'audio', 'video')

def select_all(query, items, *, tokenizer, max_length = None, ** kwargs):
    if not max_length:
        return [item for item in items if item['content_type'] not in _document_content_types], None
    
    selected, total_length = [], 0
    for item in items:
        if item.get('content_type', 'text') in _document_content_types:
            continue
        
        if 'length' not in item or not item['length']:
            item['length'] = len(tokenizer.tokenize(str(item['content'])))
        
        selected.append(item)
        total_length += item['length']
    
    return selected, total_length

select_first = select_all

def select_last(query, items, ** kwargs):
    items, length = select_all(query, items[::-1], ** kwargs)
    return items[::-1], length
