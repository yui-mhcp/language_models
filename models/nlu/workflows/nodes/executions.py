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

from .node import NodeWrapper

class SequentialExecution(NodeWrapper):
    def run(self, context):
        res = None
        for node in self.nodes:
            res = node(context)
        return res

    def plot_node(self, graph, node_id, *, shape = 'box', label = None):
        if not self.built: self.build()
        
        next_id = node_id
        g_name  = 'cluster_{}'.format(self.name)
        with graph.subgraph(name = g_name) as sub_graph:
            sub_graph.attr(label = self.name)
            
            first_node = None
            prev_node  = None
            for node in self.nodes:
                start_node, end_node, next_id, config = node.plot_node(sub_graph, next_id)
                if prev_node is not None:
                    if isinstance(prev_node, list):
                        for n in prev_node: graph.edge(n, start_node, ** config)
                    else:
                        graph.edge(prev_node, start_node, ** config)
                else:
                    first_node = start_node

                prev_node = end_node
        
        return first_node, end_node, next_id, {'lhead' : g_name}

Graph = SequentialExecution

class ParallelExecution(NodeWrapper):
    def run(self, context):
        if len(self.nodes) == 1:
            return [self.nodes[0](context)]
        
        with ThreadPool(min(cpu_count(), len(self.nodes))) as pool:
            results = [
                pool.apply_async(node, (context, ))
                for node in self.nodes
            ]
            return [res.get() for res in results]

    def plot_node(self, graph, node_id, *, shape = 'box', label = None):
        self_init, next_id = super().plot_node(graph, node_id, label = "Parallel execution")
        self_post, next_id = super().plot_node(graph, node_id)
        
        g_name = 'cluster_{}'.format(self_id)
        with graph.subgraph(name = g_name) as sub_graph:
            for node in self.nodes:
                node_id, next_id = node.plot_node(sub_graph, next_id)
                graph.edge(self_id, node_id)
        
        return self_id, next_id
