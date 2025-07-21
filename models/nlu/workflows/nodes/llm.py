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
                 
                 name   = None,
                 output_key = None,
                 
                 ** kwargs
                ):
        super().__init__(name = name, output_key = output_key)
        
        self.model  = model
        self.method = method
        self.kwargs = kwargs
        self.mapping    = mapping
        self.source_key = source_key
        self.keep_history   = keep_history
    
    def __str__(self):
        des = super().__str__()
        if self.source_key: des += "- Input key : {}\n".format(self.source_key)
        des += "- Model ({}) : {}\n".format(self.method, getattr(self.model, 'name', self.model))
        
        return des
    
    def run(self, context):
        if self.source_key:
            if isinstance(context[self.source_key], dict):
                args, kwargs = (), {** self.kwargs, ** context[self.source_key]}
            else:
                args, kwargs = (context[self.source_key], ), self.kwargs.copy()
                if self.mapping:
                    kwargs.update({k : context[v] for k, v in self.mapping.items() if v in context})
        else:
            args, kwargs = (), {** self.kwargs, ** context}
        
        kwargs.update({
            'chat_id' : context['__graph__'].name,
            'conv_id' : self.name if self.keep_history else None,
            'new_conv'  : not self.keep_history
        })
        
        return getattr(self.model, self.method)(* args, ** kwargs)['predicted']
    
    def get_config(self):
        return {
            ** super().get_config(),
            'model' : getattr(self.model, 'name', self.model),
            'method'    : self.method,
            'mapping'   : self.mapping,
            'source_key'    : self.source_key,
            'keep_history'  : self.keep_history,
            ** kwargs
        }

class TranslationNode(LLMNode):
    def __init__(self, model, *, method = 'translate', ** kwargs):
        super().__init__(model, method = method, add_answer_start = False, ** kwargs)
    