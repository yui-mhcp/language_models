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

import re
import json

from .node import Node

class TextExtractorNode(Node):
    def __init__(self, source_key, pattern, ** kwargs):
        super().__init__(** kwargs)
        self.pattern = pattern
        self.source_key = source_key
    
    def __str__(self):
        des = super().__str__()
        if self.source_key: des += "- Input key : {}\n".format(self.source_key)
        des += "- Pattern : {}\n".format(self.pattern)
        return des
    
    def run(self, context):
        source_text = context.get(self.source_key)
        if not source_text:
            raise RuntimeError('The key `{}` is missing'.format(self.source_key))
        
        matches = re.findall(self.pattern, source_text, re.DOTALL)
        
        if not matches:
            raise ValueError('No match found for pattern {}'.format(self.pattern))
        
        return matches[0] if len(matches) == 1 else matches
    
    def get_config(self):
        return {
            ** super().get_config(),
            'pattern'   : self.pattern,
            'source_key'    : self.source_key
        }

class JSONExtractorNode(TextExtractorNode):
    def __init__(self, source_key, pattern = r'```json\n(.*?)\n```', ** kwargs):
        super().__init__(source_key, pattern, ** kwargs)
    
    def run(self, context):
        try:
            matches = super().run(context)
        except ValueError:
            text = context[self.source_key]
            matches = text[text.index('{') :].strip()
        
        if isinstance(matches, list):
            return [json.loads(m) for m in matches]
        else:
            return json.loads(matches)
        
class PythonExtractorNode(TextExtractorNode):
    def __init__(self, source_key, pattern = r'```python\n(.*?)\n```', ** kwargs):
        super().__init__(source_key, pattern, ** kwargs)
