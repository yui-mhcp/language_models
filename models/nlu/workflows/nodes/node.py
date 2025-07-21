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

import logging

from abc import ABCMeta, abstractmethod
from threading import Thread, Lock, Event

logger = logging.getLogger(__name__)

class NodeManager(ABCMeta):
    _instances = {}
    
    def __call__(cls, * args, name = None, ** kwargs):
        if name is None:
            name = cls.__name__
            if name in cls._instances:
                idx = 1
                while '{}-{}'.format(name, idx) in cls._instances:
                    idx += 1
                name = '{}-{}'.format(name, idx)
        
        if name not in cls._instances:
            try:
                cls._instances[name] = super().__call__(* args, name = name, ** kwargs)
            except Exception as e:
                logger.critical('An error occured while initializing {} : {}'.format(name, e))
                raise e

        return cls._instances[name]
    
    @staticmethod
    def get(node):
        if isinstance(node, dict):
            from . import deserialize
            
            return deserialize(node)
        elif isinstance(node, str):
            if node not in NodeManager._instances:
                raise KeyError('The node {} does not exist'.format(node))

            return NodeManager._instances[node]
        elif callable(node):
            from .function import FunctionNode
            
            return FunctionNode(node)
        else:
            raise ValueError('Unsupported node type : {}'.format(type(node)))

class Node(metaclass = NodeManager):
    def __init__(self,
                 *,
                 
                 name   = None,
                 output_key = None,
                 
                 start_fetching = None,
                 stop_fetching  = None,
                 
                 ** _
                ):
        self.name   = name
        self.output_key = output_key
        
        self.start_fetching = start_fetching
        self.stop_fetching  = stop_fetching
        
        self.built = False
        
        self._stop_fetching = False
        self._prefetch_result   = None
    
    def build(self):
        self.built = True
        
        if self.start_fetching:
            if not isinstance(self.start_fetching, list): self.start_fetching = [self.start_fetching]
            self.prefetch = [
                NodeManager.get(node) if not isinstance(node, Node) else node
                for node in self.start_fetching
            ]
        
        if self.stop_fetching:
            if not isinstance(self.stop_fetching, list): self.stop_fetching = [self.stop_fetching]
            self.stop_fetching = [
                NodeManager.get(node) if not isinstance(node, Node) else node
                for node in self.stop_fetching
            ]
    
    @property
    def is_prefetched(self):
        return self._prefetch_result is not None
    
    def __repr__(self):
        return '<{} name={}>'.format(self.__class__.__name__, self.name)

    def __str__(self):
        des = "== {} ==\n".format(self.__class__.__name__.replace('Node', ''))
        if not self.name.startswith(self.__class__.__name__):
            des += "- Name : {}\n".format(self.name)
        if self.output_key:
            des += "- Output key : {}\n".format(self.output_key)
        
        return des
    
    def __call__(self, context = None, /, ** kwargs):
        """ Start the node and returns its result only (i.e., without context) """
        return self.start(context, ** kwargs)[1]
    
    def prefetch(self, context, /):
        if self.is_prefetched: return
        
        def run_and_set_result(context):
            result = self.run(context)
            self._prefetch_result(result)
        
        self._stop_fetching = False
        self._prefetch_result   = AsyncResult()

        Thread(target = run_and_set_result, args = (context, ), daemon = True).start()
    
    def stop_prefetch(self):
        self._stop_fetching = True
    
    @abstractmethod
    def run(self, context, /):
        """ Execute the node logic, and returns its result """
    
    def start(self, context = None, /, ** kwargs):
        """ Initialize and execute the node, and returns the updated context and the node result """
        if not self.built: self.build()
        
        if context is None: context = kwargs
        _pop_graph = '__graph__' not in context
        if _pop_graph: context['__graph__'] = self
        
        if context.get('__abort__', False):
            logger.warning('The workflow is stopped, skipping {}'.format(self))
            return context, None
        
        self.on_start(context)
        
        if self.is_prefetched:
            result = self._prefetch_result.get()
            self._prefetch_result = None
        else:
            result = self.run(context)

        self.on_finished(context, result)
        
        if _pop_graph: context.pop('__graph__')
        
        return context, result

    def on_start(self, context):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Start node {}'.format(self.name))
    
        if self.start_fetching:
            for node in self.start_fetching: node.prefetch()
        
        if self.stop_fetching:
            for node in self.stop_fetching: node.stop_prefetch()

    def on_finished(self, context, result):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Finished node {}'.format(self.name))
        
        if self.output_key:
            context[self.output_key] = result
    
    def get_config(self):
        if not self.built: self.build()
        
        return {
            'class_name' : self.__class__.__name__,
            'name'       : self.name,
            'output_key' : self.output_key
        }

    def plot(self,
             filename = None,
             name = None,
             view = False,
             graph = None,
             node_graph = None,
             node_id = 0,
             node_name = None,
             str_id = None
            ):
        """ Builds a `graphviz.Digraph` representing the workflow """
        import graphviz as gv
        
        if graph is None:
            if name is None: name = filename if filename else self.name
            graph = gv.Digraph(name = name, filename = filename)
            graph.attr(compound = 'true')
        

        self.plot_node(graph, 0)
        
        if view or filename is not None:
            basename, _, format = filename.rpartition('.') if filename else (None, None, 'pdf')
            graph.render(
                filename = basename, view = view, cleanup = True, format = format
            )
        
        return graph

    def plot_node(self, graph, node_id, *, shape = 'box', label = None):
        if not self.built: self.build()
        if not label: label = str(self)
        
        str_id = str(node_id)
        
        graph.node(str_id, id = str_id, shape = shape, label = label)
        
        return str_id, str_id, node_id + 1, {}

class NodeWrapper(Node):
    def __init__(self, * args, nodes = None, ** kwargs):
        super().__init__(** kwargs)

        self._nodes = list(nodes if nodes else args)
    
    @property
    def nodes(self):
        return self._nodes

    def build(self):
        super().build()
        
        for i, node in enumerate(self._nodes):
            if not isinstance(node, Node):
                self._nodes[i] = NodeManager.get(node)
            

    def get_config(self):
        return {
            ** super().get_config(),
            'nodes' : [node.get_config() for node in self.nodes]
        }
    
    
    