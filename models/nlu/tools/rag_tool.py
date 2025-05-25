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
import math

from typing import Union

from .tool import Tool
from ..prompts import dedent, prompt_docstring

@prompt_docstring(
    en = "Perform semantic search of `query` and return the `k` most relevant passages in `documents`.",
    fr = dedent("""
    Retourne les `k` paragraphes les plus pertinents dans les `documents` fournis.
    
    Arguments :
        - query : une (liste d') information(s) à rechercher
        - documents : une liste de document(s) (.pdf, .docx, .txt, ...) à utiliser
    Return :
        - paragraphs    : une liste de `dict` avec la clef `text`
    
    Exemple :
    ```python
    for paragraph in rag(query, documents):
        print(paragraph['text'])
    ```
    """)
)
def rag(query   : Union[str, list],
        documents   : list,
        *,
        
        k   = 5,
        reverse = True,
        retriever   = None
       ):
    if isinstance(query, str): query = [query]
    
    if isinstance(retriever, str):
        from models import get_pretrained
        retriever = get_pretrained(retriever)

    db  = retriever.predict(documents = documents, save = False)
    retrieved = retriever.retrieve(
        query, db, reverse = reverse, k = max(2,int(math.ceil(k / len(query)))), run_eagerly = True
    )
    
    results = []
    for res in retrieved:
        results.extend(res)
    return results

RAGTool = Tool.from_function(rag, ignore = ['retriever', 'reverse'])