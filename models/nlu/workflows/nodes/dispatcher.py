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

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from .node import Node

class DispatcherNode(Node):
    def __init__(self, branches, ** kwargs):
        super().__init__(** kwargs)
        
        self.branches = branches

    def build(self):
        super().build()
        
        if isinstance(self.branches, list):
            self.branches = {k : v for k, v in self.branches}
        
        for k, node in self.branches.items():
            if not isinstance(node, Node):
                self.branches[k] = NodeManager.get(node)
    

    def run(self, context):
        branches = {k : v for k, v in self.branches.items() if context.get(k)}
        
        if len(branches) == 0:
            return []
        elif len(branches) == 1:
            return {k : v(context) for k, v in branches.items()}
        
        with ThreadPool(min(cpu_count(), len(self.nodes))) as pool:
            results = {
                k : pool.apply_async(v, (context, )) for k, v in branches.items()
            }
            return {k : v.get() for k, v in results.items()}

    def plot_node(self, graph, node_id, *, shape = 'box', label = None):
        if not self.built: self.build()
        self_id, next_id = super().plot_node(graph, node_id, shape = 'elipse')
        
        for key, node in self.branches.items():
            node_id, next_id = node.plot(graph, next_id)
            graph.edge(self_id, node_id, label = str(key))
        
        return self_id, next_id

    def get_config(self):
        return {
            ** super().get_config(),
            'condition' : self.condition.get_config(),
            'branches'  : [[k, v.get_config()] for k, v in self.branches.items()]
        }
