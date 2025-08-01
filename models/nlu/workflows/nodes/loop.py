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

from .node import NodeManager, Node

MAXIMUM_ITERATION = 1024

class LoopNode(Node):
    def __init__(self, body, cond = None, *, max_iter = MAXIMUM_ITERATION, ** kwargs):
        assert cond or max_iter
        
        super().__init__(** kwargs)
        
        self.body   = body
        self.cond   = cond
        self.max_iter   = max_iter
    
    def build(self):
        super().build()
        
        if isinstance(self.cond, str):
            from .value import ContextValueNode
            
            self.cond = ContextValueNode(self.condition)
        elif self.cond is None:
            from .value import ValueNode
            
            self.cond = ValueNode(True)
        elif not isinstance(self.cond, Node):
            self.cond = NodeManager.get(self.cond)
        
        if not isinstance(self.body, Node):
            self.body = NodeManager.get(self.body)

    @property
    def nested_nodes(self):
        return [self.cond, self.body]
    
    def run(self, context):
        res, iteration = None, 0
        while self.cond(context) and iteration < self.max_iter and not self.is_stopped():
            res = self.body(context)
            iteration += 1
            
        return res
    
    def plot_node(self, graph, node_id, *, shape = 'box', label = None):
        if not self.built: self.build()
        
        cond_id, _, next_id, _ = self.cond.plot_node(graph, node_id, shape = 'diamond')
        body_id, last_id, next_id, config = self.body.plot_node(graph, next_id)

        graph.edge(cond_id, body_id, label = 'True', ** config)
        for n in (last_id if isinstance(last_id, list) else [last_id]):
            graph.edge(n, cond_id, label = 'Loop')
        
        return cond_id, cond_id, next_id, {}

    def get_config(self):
        return {
            ** super().get_config(),
            'body'  : self.body.get_config(),
            'cond'  : self.cond.get_config(),
            'max_iter'  : self.max_iter
        }
