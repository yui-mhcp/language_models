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

class ValueNode(Node):
    def __init__(self, value, output_key, ** kwargs):
        super().__init__(output_key = output_key, ** kwargs)
        self.value = value
    
    def __str__(self):
        return super().__str__() + "- Value : {}\n".format(self.value)
    
    def run(self, context):
        return self.value
    
    def get_config(self):
        return {** super().get_config(), 'value' : self.value}
    
class ContextValueNode(Node):
    def __init__(self, key, ** kwargs):
        super().__init__(** kwargs)
        self.key = key
    
    def __str__(self):
        return super().__str__() + "- Key : {}\n".format(self.key)

    def run(self, context):
        return context[self.key]
    
    def get_config(self):
        return {** super().get_config(), 'key' : self.key}
    
