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

from .naive_selectors import select_all

logger = logging.getLogger(__name__)

def _base_rag_selector(embedding_model,
                       query,
                       *,

                       texts    = None,
                       documents    = None,

                       k    = 10,
                       reverse  = True,
                       threshold    = 0.,
                       tokenizer    = None,
                       max_length   = None,

                       ** kwargs
                      ):
    if isinstance(embedding_model, str):
        from models import get_pretrained
        embedding_model = get_pretrained(embedding_model)

    from models.encoder.rag import index_data, search

    # persistent database : already-indexed data is skipped at each call (`hash` deduplication)
    database = os.path.join(embedding_model.pred_dir, 'rag_database')
    database = index_data(
        embedding_model, database, texts = texts, documents = documents, ** kwargs
    )
    items   = search(embedding_model, database, query, k = k)
    if threshold: items = [it for it in items if it['score'] >= threshold]
    if reverse:   items = items[::-1]

    if items and logger.isEnabledFor(logging.DEBUG):
        logger.debug('{} items retrieved with scores in [{:.3f}, {:.3f}] range'.format(
            len(items), items[0]['score'], items[-1]['score']
        ))

    return select_all(query, items, max_length = max_length, tokenizer = tokenizer)

def select_rag(query, items, *, embedding_model, documents = None, ** kwargs):
    if not documents: kwargs['texts'] = items
    return _base_rag_selector(embedding_model, query, documents = documents, ** kwargs)

def select_rag_messages(query, items, *, embedding_model, skip_last = 0, ** kwargs):
    if skip_last:
        if skip_last >= len(items): return [], 0
        items = items[:-skip_last]

    from utils.text import paragraph_to_text

    # a `Conversation` is unknown to the encoder : messages are converted to canonical
    # texts here, `source` being the conversation id
    texts = []
    for msg in items:
        text = paragraph_to_text(msg)
        if text:
            texts.append({** msg, 'text' : text, 'source' : msg.get('conv_id', 'messages')})

    if not texts: return [], 0

    return _base_rag_selector(embedding_model, query, texts = texts, ** kwargs)
