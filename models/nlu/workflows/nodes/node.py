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
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

logger = logging.getLogger(__name__)

class NodeManager(ABCMeta):
    _pools  = {}
    _instances  = {}
    
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
        if isinstance(node, Node):
            return node
        elif isinstance(node, list):
            return [NodeManager.get(n) for n in node]
        elif isinstance(node, dict):
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

    @staticmethod
    def run_async(node, context, /, *, max_workers = None, callback = None, ** kwargs):
        root = context['__graph__']
        if root not in NodeManager._pools:
            if max_workers is None: max_workers = cpu_count()
            NodeManager._pools[root] = ThreadPool(max_workers).__enter__()
        
        return NodeManager._pools[root].apply_async(node, (context, ), kwargs, callback = callback)
    
    @staticmethod
    def finalize(root, context):
        root = context.pop('__graph__')
        
        if root in NodeManager._pools:
            NodeManager._pools.pop(root).__exit__(None, None, None)

        
class Node(metaclass = NodeManager):
    """
        A `Node` object is the base component of any workflow.
        A `workflow` is a graph-like structure composed of `Node`s.
        Each `Node` can be atomic or nested (i.e., contains `Node`) for more complex logic.
        
        The major difference with other "agent" libraries, is that a `Node` can be used as a standalone workflow. This means that each sub-elements from a workflow can be extracted and executed as is.
        
        The `Node` class is an abstraction : the `run(context)` method has to be defined in sub-classes
        
        The `run` method may also implement an interruption mechanism, in case of `is_stopped()` returns `True`. This can happen in two different scenarios :
        1) The `abort` method is called, aborting the execution of the graph itself
        2) The `cancel` method is called (after a `prefetch` call), cancelling the prefetching of the node
        The `nested_nodes` property may be defined to also abort/cancel sub-nodes execution.
        
        
        The `prefetch` mechanism enables to pre-compute the node result in a separate thread, to directly return the result once the `start` method is called. 
        This is especially useful to pre-compute most probable paths in a workflow.
        
        Example :
        ```python
        graph = Graph(
            # asks to a LLM whether we should search on web or not
            LLMNode(..., start_prefetch = 'final_answer')
            ConditionNode(
                ShouldSearchOnWeb,
                # Performs the web search
                true_node = WebNode(..., stop_prefetch = 'final_answer')
            ),
            # Generates the final answer
            LLMNode(..., name = 'final_answer')
        )
        ```
        In this case, the majority of queries will not require a web search, while asking to the LLM whether or not we should search is a time-consuming task.
        The solution is, just before executing the 1st node, start the final_answer node in a separate thread.
        Therefore, during the 1st node execution, the final answer will be generated while supposing the query will not require a web search.
        On the other hand, if the `WebNode` if called (i.e., the query requires a web search), the final answer will be cancelled (`cancel`), and re-computed with the web results.
        
        **WARNING** This behavior can save significant amount of time to generate the final answer when no web search is required, but leads to useless computation if search is required. It is therefore critical to only prefetch paths where speed is required.
    """
    def __init__(self,
                 *,
                 
                 name   = None,
                 output_key = None,
                 
                 start_prefetch = None,
                 stop_prefetch  = None,
                 
                 ** kwargs
                ):
        self.name   = name
        self.kwargs = kwargs
        self.output_key = output_key
        
        self.start_prefetch = start_prefetch if start_prefetch is not None else []
        self.stop_prefetch  = stop_prefetch if stop_prefetch is not None else []
        
        self.built = False
        
        self._stopper   = lambda: False
        self._aborted   = False
        self._cancelled = False
        self._prefetch_result   = None
    
    def build(self):
        """ Builds the node, e.g., by normalizing sub-nodes with `NodeManager.get` """
        self.built = True
        
        if self.start_prefetch:
            if not isinstance(self.start_prefetch, list): self.start_prefetch = [self.start_prefetch]
            self.start_prefetch = NodeManager.get(self.start_prefetch)
        
        if self.stop_prefetch:
            if not isinstance(self.stop_prefetch, list): self.stop_prefetch = [self.stop_prefetch]
            self.stop_prefetch = NodeManager.get(self.stop_prefetch)
    
    @property
    def nested_nodes(self):
        return []
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.name == other.name
    
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
        """
            Start the node and returns its result only (i.e., without context)
            This is therefore equivalent to `result = node.start(context)[1]`
        """
        return self.start(context, ** kwargs)[1]
    
    @abstractmethod
    def run(self, context, /, ** kwargs):
        """ Execute the node logic, and returns its result """

    def abort(self):
        """ Stop the node execution """
        self._abort = True
        
        if self.built:
            for node in self.nested_nodes: node.abort()

        self.on_stop()

    def cancel(self):
        """
            Stop the node prefetching
            In practice, this function :
            1) Set `cancelled` to `True`
            2) Call `self.stop()`
            3) Wait until the async result is available (i.e., the `run` is finished)
            4) Set `cancelled` to `False` and `prefetch_result` to `None`
            This enables the node to be re-fetched after being interrupted
            
            It is therefore important to define a stopping behavior in the `run` method to enable as fast as possible interruption.
        """
        self._cancelled = True
        
        if self.built:
            for node in self.nested_nodes: node.cancel()
        
        self.on_stop()
        
        self._prefetch_result.get()
        self._prefetch_result   = None
        self._cancelled = False

    def clone(self):
        """ Return a new instance of the node with same config but different name """
        return self.__class__(** {** self.get_config(), 'name' : None})

    def is_aborted(self):
        """ Return whether or not the node is aborted (i.e., `abort()` has been called) """
        return self._aborted

    def is_cancelled(self):
        """ Return whether or not the node is cancelled (i.e., `cancel()` has been called) """
        return self._cancelled
    
    def is_prefetched(self):
        """ Return whether or not the node is prefetched (i.e., `prefetch()` has been called) """
        return self._prefetch_result is not None
    
    def is_stopped(self):
        """ Return whether or not the node should stop (either `abort` or `cancel` has been called) """
        return self.is_aborted() or self.is_cancelled() or self._stopper()
    
    def on_start(self, context):
        """ Callback method called before running the node """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Start node {}'.format(self.name))
    
        if self.start_prefetch:
            for node in self.start_prefetch: node.prefetch()
        
        if self.stop_prefetch:
            for node in self.stop_prefetch: node.abort()

    def on_stop(self):
        """ Callback method called when the node is interrupted (either `abort` either `cancel`) """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Node {} is {}'.format(
                self.name, 'aborted' if self.is_aborted() else 'cancelled'
            ))

    def on_finished(self, context, result):
        """ Callback method called just after the node has been executed """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Finished node {}'.format(self.name))
        
        if self.output_key:
            context[self.output_key] = result

    def prefetch(self, context, /, ** kwargs):
        """
            Prepare the node result by calling `self.run_async` without effectively starting the node (i.e., `on_start` is not called).
            
            **IMPORTANT** : This method is only defined for "atomic" nodes (i.e., nodes that do not have nested nodes). In the case of non-atomic nodes, calling `self.run` will result in calling the `start` method of sub-nodes, which is unexpected for the `prefetch` mechanism.
            
            Please overwrite the method to call the "prefetch" method only from the expected sub-nodes.
        """
        if self.is_prefetched() or self.is_aborted(): return
        if not self.built: self.build()
        
        if self.is_cancelled():
            raise RuntimeError("The node is prefetched while cancelled, which should not happen")
        elif self.nested_nodes:
            raise NotImplementedError("The `prefetch` method should only be called on atomic nodes, but {} has nested nodes. Please overwrite the method to prefetch the expected sub nodes.".format(self))
        
        self._prefetch_result   = NodeManager.run_async(self.run, context, ** kwargs)
    
    def start(self, context = None, /, _stopper = None, ** kwargs):
        """
            Initialize and execute the node. Then returns the updated context and the node result
            
            Arguments :
                - context   : a `dict` containing the graph variables / state
                - _stopper  : a `callable` that returns `True` if the node should be stopped
                - kwargs    : additional kwargs forwarded to `run`, or used as context if it is the root of the workflow
            Return :
                - context   : the (possibly in-place updated) context
                - result    : the node execution result (i.e., the output of `self.run`)
            
            Note : if no context is provided, `kwargs` is used as context
        """
        if context is None: context, kwargs = kwargs, {}

        if _stopper is not None:
            if _stopper(): return
            self._stopper       = _stopper
            if self.nested_nodes: kwargs['_stopper'] = _stopper
        
        if not self.built: self.build()
        
        _is_root = '__graph__' not in context
        if _is_root: context['__graph__'] = self
        
        if context.get('__abort__', False) or self.is_aborted():
            logger.warning('The workflow is stopped, skipping {}'.format(self))
            if _is_root: NodeManager.finalize(self, context)
            return context, None
        
        self.on_start(context)
        
        result = None
        try:
            if self.is_prefetched():
                result = self._prefetch_result.get()
                self._prefetch_result = None
            else:
                result = self.run(context, ** kwargs)
        finally:
            self.on_finished(context, result)

            if _is_root: NodeManager.finalize(self, context)
        
        return context, result
    
    def get_config(self):
        if not self.built: self.build()
        
        return {
            'class_name' : self.__class__.__name__,
            'name'       : self.name,
            'output_key' : self.output_key,
            'start_prefetch'    : [node.name for node in self.start_prefetch],
            'stop_prefetch'     : [node.name for node in self.stop_prefetch],
            ** self.kwargs
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

    @property
    def nested_nodes(self):
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
    
    
    