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

def select_rag(query,
               items,
               *,
               
               embedding_model,

               path = None,
               directory    = None,
               documents    = None,
               
               chunk_size   = 256,
               group_by     = 'filename',
               
               k    = 10,
               threshold = 0.45,
               
               conv = None,
               tokenizer    = None,
               max_length   = None,
               
               ** kwargs
              ):
    if isinstance(embedding_model, str):
        from models import get_pretrained
        embedding_model = get_pretrained(embedding_model)
    
    if not path: path = 'documents.db' if documents else 'messages.db'
    
    inp = items if not documents else None
    
    db = embedding_model.predict(
        items if documents is None else None,
        
        chunk_size  = chunk_size,
        group_by    = group_by,
        
        path    = path,
        directory   = directory,
        documents   = documents,
        ** kwargs
    )
    res = embedding_model.retrieve(query, db, k = k)[0]
    filtered = [r for r in res if r['score'] > threshold]
    print('# paragraphs found : {} [{:.3f}, {:.3f}] - filtered : {}'.format(
        len(res), res[-1]['score'], res[0]['score'], len(filtered)
    ))
    return filtered, 0
