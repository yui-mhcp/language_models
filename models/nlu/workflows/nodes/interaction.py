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

class PrintNode(Node):
    def __init__(self, source_key, ** kwargs):
        super().__init__(** kwargs)
        self.source_key = source_key
    
    def __str__(self):
        return super().__str__() + '- Input key : {}'.format(self.source_key)
    
    def run(self, context):
        print(context[self.source_key])
        return None
    
    def get_config(self):
        return {** super().get_config(), 'source_key' : self.source_key}

class CLINode(Node):
    def __init__(self, prompt = '', ** kwargs):
        super().__init__(** kwargs)
        self.prompt = prompt
    
    def run(self, context):
        return input(self.prompt)

    def get_config(self):
        return {** super().get_config(), 'prompt' : self.prompt}
