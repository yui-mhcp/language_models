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

class LLMNode(Node):
    def __init__(self,
                 model,
                 *,
                 
                 method = 'answer',
                 mapping    = None,
                 source_key = None,
                 keep_history   = True,
                 
                 ** kwargs
                ):
        super().__init__(** kwargs)
        
        self.model  = model
        self.method = method
        self.mapping    = mapping
        self.source_key = source_key
        self.keep_history   = keep_history
    
    def build(self):
        super().build()
        
        self.request_manager = RequestManager(self)
    
    def __str__(self):
        des = super().__str__()
        if self.source_key: des += "- Input key : {}\n".format(self.source_key)
        des += "- Model  : {}\n".format(getattr(self.model, 'name', self.model))
        des += "- Method : {}\n".format(self.method)
        
        return des
    
    def run(self, context, ** kwargs):
        if isinstance(self.model, str):
            from models import get_pretrained
            
            self.model = get_pretrained(model)

        if self.source_key:
            if isinstance(context[self.source_key], dict):
                args, kwargs = (), {** self.kwargs, ** context[self.source_key]}
            else:
                args, kwargs = (context[self.source_key], ), self.kwargs.copy()
                if self.mapping:
                    kwargs.update({
                        key : context[ctx_key] for ctx_key, key in self.mapping.items()
                        if ctx_key in context
                    })
        else:
            args, kwargs = (), {** self.kwargs, ** context}
        
        if self.keep_history:
            kwargs['conv_id'] = self.name
        else:
            kwargs['messages'] = []
        
        if 'request_manager' in kwargs:
            raise NotImplementedError('The `request_manager` is currently not supported')
        
        kwargs['request_manager'] = self.request_manager
        
        return getattr(self.model, self.method)(* args, ** kwargs)['predicted']
    
    def get_config(self):
        return {
            ** super().get_config(),
            'model' : getattr(self.model, 'name', self.model),
            'method'    : self.method,
            'mapping'   : self.mapping,
            'source_key'    : self.source_key,
            'keep_history'  : self.keep_history
        }

class TranslationNode(LLMNode):
    def __init__(self, model, *, method = 'translate', ** kwargs):
        super().__init__(model, method = method, add_answer_start = False, ** kwargs)

class RequestManager:
    def __init__(self, node):
        self.node = node
    
    def is_aborted(self, request_id = None):
        return self.node.is_stopped()
    
    def __call__(self, item, request_id = None):
        return not self.is_aborted()
