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

class BranchingNode(Node):
    def __init__(self, condition, branches, ** kwargs):
        super().__init__(** kwargs)
        
        self.condition  = condition
        self.branches   = branches
    
    def build(self):
        super().build()
        
        if isinstance(self.condition, str):
            from .value import ContextValueNode
            
            self.condition = ContextValueNode(self.condition)
        elif not isinstance(self.condition, Node):
            self.condition = NodeManager.get(self.condition)
        
        
        if isinstance(self.branches, list):
            self.branches = {k : v for k, v in self.branches}
        
        for k, node in self.branches.items():
            if not isinstance(node, Node):
                self.branches[k] = NodeManager.get(node)
    
    @property
    def nested_nodes(self):
        return [self.condition] + list(self.branches.values())
    
    def run(self, context, ** kwargs):
        value = self.condition(context, ** kwargs)

        if value in self.branches:
            return self.branches[value](context, ** kwargs)
        elif 'default' in self.branches:
            return self.branches['default'](context, ** kwargs)
        else:
            return None
    
    def plot_node(self, graph, node_id, *, shape = 'box', label = None):
        if not self.built: self.build()
        self_id, next_id = self.condition.plot_node(graph, node_id, shape = 'diamond')
        
        branches = []
        for key, node in self.branches.items():
            node_id, _, next_id = node.plot(graph, next_id)
            graph.edge(self_id, node_id, label = str(key))
            branches.append(node_id)
        
        return self_id, branches, next_id, {}

    def get_config(self):
        return {
            ** super().get_config(),
            'condition' : self.condition.get_config(),
            'branches'  : [[k, v.get_config()] for k, v in self.branches.items()]
        }

class ConditionNode(Node):
    def __init__(self, condition, true_node, false_node = None, ** kwargs):
        super().__init__(** kwargs)
        
        self.condition  = condition
        self.true_node  = true_node
        self.false_node = false_node

    def build(self):
        super().build()
        
        if isinstance(self.condition, str):
            from .value import ContextValueNode
            
            self.condition = ContextValueNode(self.condition)
        elif not isinstance(self.condition, Node):
            self.condition = NodeManager.get(self.condition)
        
        if not isinstance(self.true_node, Node):
            self.true_node = NodeManager.get(self.true_node)
        
        if self.false_node is not None and not isinstance(self.false_node, Node):
            self.false_node = NodeManager.get(self.false_node)
    
    @property
    def nested_nodes(self):
        return [self.condition, self.true_node] + ([] if self.false_node is None else [self.false_node])

    def run(self, context, ** kwargs):
        value = self.condition(context, ** kwargs)

        if self.is_stopped():
            return None
        elif value:
            return self.true_node(context, ** kwargs)
        elif self.false_node is not None:
            return self.false_node(context, ** kwargs)
        else:
            return None
    
    def plot_node(self, graph, node_id, *, shape = 'box', label = None):
        if not self.built: self.build()
        self_id, cond_id, next_id, _ = self.condition.plot_node(graph, node_id, shape = 'diamond')
        
        true_id, last_true_id, next_id, config = self.true_node.plot_node(graph, next_id)
        graph.edge(cond_id, true_id, label = 'True', ** config)
        branches = [last_true_id] if not isinstance(last_true_id, list) else last_true_id
        if self.false_node is not None:
            false_id, last_false_id, next_id, config = self.false_node.plot_node(graph, next_id)
            graph.edge(cond_id, false_id, label = 'False', ** config)
            branches.extend(
                last_false_id if isinstance(last_false_id, list) else [last_false_id]
            )
        else:
            branches.append(cond_id)

        return self_id, branches, next_id, {}

    def get_config(self):
        return {
            ** super().get_config(),
            'condition' : self.condition.get_config(),
            'true_node' : self.true_node.get_config(),
            'false_node'    : self.false_node.get_config() if self.false_node is not None else None
        }
