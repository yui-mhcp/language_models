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

from .node import Node

from utils.text.web import search_on_web, process_urls
from utils.text.parsers.html_parser import get_link

class WebNode(Node):
    def __init__(self, source_key, ** kwargs):
        super().__init__(** kwargs)
        self.source_key = source_key
    
    def __str__(self):
        return super().__str__() + "- Input key : {}\n".format(self.source_key)
    
    def run(self, context):
        return self._run(** context[self.source_key])
    
    def _run(self, *, url = None, query = None, link_id = None, n = 5):
        urls = []
        if url:                 urls.append(url)
        if link_id is not None: urls.append(get_link(link_id))
        if query:               urls.extend(search_on_web(query, parse = False, n = n))

        res = process_urls(urls, track_href = True, n = n)
        
        return list(res.values())
    
    def get_config(self):
        return {** super().get_config(), 'source_key' : self.source_key}
