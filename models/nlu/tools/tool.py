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

import inspect

from functools import cached_property
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional, Union

from ..prompts import Prompt, get_translation

_globals_ignore = {'self', 'model'}

@dataclass
class Tool:
    name    : str
    description : Union[Prompt, Dict, str]
    function    : Callable  = field(repr = False)
    instructs   : List  = field(default_factory = list, repr = False)
    ignore      : List  = field(default_factory = list, repr = False)
    metadata    : Dict[str, Any] = field(default_factory = dict, repr = False)
    
    def __post_init__(self):
        if self.function is None:
            self.function = self.invoke
    
    @cached_property
    def argnames(self):
        return set(list(inspect.signature(self.function).parameters.keys()))
    
    @cached_property
    def signature(self):
        return inspect.Signature([
            p for name, p in inspect.signature(self.function).parameters.items()
            if name not in _globals_ignore and name not in self.ignore
        ])
    
    def __call__(self, * args, ** kwargs):
        if 'kwargs' not in self.argnames:
            kwargs = {k : v for k, v in kwargs.items() if k in self.argnames}
        return self.function(* args, ** kwargs)
    
    def to_signature(self, lang = 'en'):
        return "def {}{}:\n    {}".format(
            self.name, self.signature,
            get_translation(self.description, lang).replace('\n', '\n    ').strip()
        )

    def get_instructions(self, lang = 'en'):
        return [get_translation(instruct) for instruct in self.instructions]
    
    def to_json(self, lang = 'en'):
        return {
            "name"  : self.name,
            "description"   : get_translation(self.description, lang),
            "properties"    : {
                name : {"name" : name}
                for name, param in self.signature.parameters.items()
            },
            "required": [
                name for name, param in self.signature.parameters.items()
                if param.default == inspect._empty
            ]
        }
    
    @classmethod
    def from_function(cls, function, ** kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = getattr(function, '__name__', function.__class__.__name__)
        if 'description' not in kwargs:
            if hasattr(function, 'prompts'):
                kwargs['description'] = function.prompts
            else:
                kwargs['description'] = getattr(function, '__doc__', '').strip()
        return cls(function = function, ** kwargs)