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

import re
import inspect
import importlib

from .node import Node

_lambda_pattern = r'(lambda.*)'

class FunctionNode(Node):
    def __init__(self, func, source_key = None, ** kwargs):
        super().__init__(** kwargs)
        
        self.func = func
        self.source_key = source_key
    
    @property
    def str_fn(self):
        if self.func.__name__ == '<lambda>':
            lambda_fn = re.search(
                _lambda_pattern, inspect.getsource(self.func), flags = re.DOTALL
            ).group(1).strip().replace('"', r'\"')
            if lambda_fn[-1] == ',': lambda_fn = lambda_fn[:-1]
            return lambda_fn
        else:
            return '{}.{}'.format(self.func.__module__, self.func.__name__)

    def __str__(self):
        des = super().__str__()
        
        if self.source_key: des += "- Input key : {}\n".format(self.source_key)
        des += "- Function : {}\n".format(self.str_fn)
        return des

    def build(self):
        super().build()
        
        if isinstance(self.func, str):
            if self.func.startswith('lambda'):
                exec('self.func = {}'.format(self.func.replace(r'\"', '"')))
            else:
                module, _, name = self.func.rpartition('.')
                self.func = getattr(importlib.import_module(module), name)
    
    def run(self, context):
        return self.func(context if not  self.source_key else context[self.source_key])
    
    def get_config(self):
        return {
            ** super().get_config(),
            'func'  : self.str_fn,
            'source_key' : self.source_key
        }
