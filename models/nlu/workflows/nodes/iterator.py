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

class IteratorNode(Node):
    def __init__(self, body, iterable, item_key, ** kwargs):
        super().__init__(** kwargs)
        
        self.body   = body
        self.item_key   = item_key
        self.iterable   = iterable
        
    def __new__(cls, * args, parallel = None, ** kwargs):
        if cls is not IteratorNode:
            return super().__new__(cls)
        elif parallel is None:
            raise RuntimeError('You must specify whether the iterator should be applied in parallel or not')
        elif parallel:
            return ParallelIteratorNode(* args, ** kwargs)
        else:
            return SequentialIteratorNode(* args, ** kwargs)
    
    def build(self):
        super().build()
        
        if not isinstance(self.iterable, Node):
            from .value import ContextValueNode, ValueNode
            
            if isinstance(self.iterable, str):
                self.iterable = ContextValueNode(self.iterable)
            else:
                self.iterable = ValueNode(self.iterable)
        
        if not isinstance(self.body, Node):
            self.body = NodeManager.get(self.body)
    
    def plot_node(self, graph, node_id, *, shape = 'box', label = None):
        if not self.built: self.build()
        
        iter_id, _, next_id, _ = self.iterable.plot_node(graph, node_id, shape = 'circle')
        body_id, last_id, next_id, config = self.body.plot_node(graph, next_id)

        if isinstance(self, SequentialIteratorNode):
            graph.node(str(next_id), label = 'item #i')
            
            graph.edge(iter_id, str(next_id))
            graph.edge(str(next_id), body_id, ** config)
            for n in (last_id if isinstance(last_id, list) else [last_id]):
                graph.edge(n, iter_id, label = 'Loop')
            next_id += 1
        else:
            for i in range(3):
                graph.node(str(next_id), label = 'item #{}'.format(
                    '...' if i == 2 else i + 1
                ))
                graph.edge(iter_id, str(next_id))
                graph.edge(str(next_id), body_id, ** config)
                next_id += 1
        
        return iter_id, last_id, next_id, {}
    
    def get_config(self):
        return {
            ** super().get_config(),
            'condition' : self.condition.get_config(),
            'item_key'  : self.item_key,
            'body'  : self.body.get_config()
        }

class SequentialIteratorNode(IteratorNode):
    def run(self, context):
        res = None
        for item in self.iterable(context):
            context[self.item_key] = item
            context, res = self.body(context)
        
        return res

class ParallelIteratorNode(IteratorNode):
    def run(self, context):
        iterable = self.iterable(context)
        
        if len(iterable) == 0:
            return []
        elif len(iterable) == 1:
            context[self.item_key] = iterable[0]
            return [self.body(context)]
        
        with ThreadPool(min(cpu_count(), len(iterable))) as pool:
            outputs = [
                pool.apply_async(self.body.start, ({** context, self.item_key : item}, ))
                for item in iterable
            ]
            outputs = [out.get() for out in outputs]
        
        results = []
        for ctx, res in outputs:
            context.update(ctx)
            results.append(res)
        
        return results
        
