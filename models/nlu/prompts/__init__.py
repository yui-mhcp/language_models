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

import os
import glob
import importlib

from functools import wraps

from .prompt import Prompt, get_prompt, dedent
from .prompt_formatter import PromptFormatter

def set_prompt(prompt, task = None, key = None, lang = None):
    """ Set a new prompt for the given task, prompt key and lang """
    if not task:
        for task, p in prompt.items():
            set_prompt(p, task, key, lang)
    elif key is None:
        for key, p in prompt.items():
            set_prompt(p, task, key, lang)
    elif lang is None:
        for lang, p in prompt.items():
            set_prompt(p, task, key, lang)
    else:
        _prompts.setdefault(task, {}).setdefault(key, Prompt()).update({lang : prompt})

_prompts = {}
for file in glob.glob(os.path.join(* __package__.split('.'), 'prompts_*.py')):
    module = importlib.import_module(file.replace(os.path.sep, '.')[:-3])
    
    lang = file[:-3].split('_')[-1]
    for k, v in vars(module).items():
        if k.startswith('prompts_'):
            task = k[8:]
            for prompt_key, text in v.items():
                set_prompt(text, task, prompt_key, lang)


def get_prompts(task, lang = None, *, prompts = None):
    """
        Return a mapping `{prompt_key : text}` associated to the given `task` and `lang`
        
        Arguments :
            - task  : a task representing a valid entry in `_prompts`
            - lang  : the prompt language (if multiple translations are available)
            - prompts   : a nested mapping `{task : {prompt_key : {lang : prompt}}}`
                          If not provided, default to `_prompts`
        Return :
            - prompts   : a mapping `{prompt_key : prompt}` for the given task / lang
                          if `task` is not in `prompts`, return an empty `dict`
    """
    if prompts is None:
        global _prompts
        prompts = _prompts
    
    if task not in prompts: return {}
    prompts = prompts[task]
    
    return {k : get_prompt(v, lang) for k, v in prompts.items()}

def add_prompt_wrapper(_task, /, *, fn = None):
    """
        Wraps a function to automatically add all `task`-related prompts to `kwargs`
        
        Example :
        ```
        @add_prompt_wrapper('translate')
        def translate(self, text, system_prompt, ...):
            return self.infer(system_prompt = system_prompt, ...)
        
        print(model.translate(text)) # this works as `system_prompt` will automatically be added to kwargs
        print(model.translate(text, system_prompt = '...')) # specify a custom system prompt instead of the default one
        ```
        
        Adds the `prompt_task` attribute to the wrapped function
    """
    def wrapper(fn):
        @wraps(fn)
        def inner(self,
                  * args,
                  
                  task  = None,
                  lang  = None,
                  
                  _add_prompts  = True,
                  
                  ** kwargs
                 ):
            if _add_prompts:
                if not lang: lang = self.lang
                
                config = {}
                for t in (_task, task):
                    if t: config.update(get_prompts(t, lang))
                
                kwargs = {** config, ** kwargs}
            
            return fn(self, * args, lang = lang, ** kwargs)
        
        inner.prompt_task = _task
        return inner
    
    return wrapper if fn is None else wrapper(fn)

def prompt_docstring(prompt = None, /, ** kwargs):
    """
        Setup a multilingual docstring
        This is especially useful for tools description if the model should be multilingual
        
        Example :
        ```
        @prompt_docstring(
            en  = 'Multiplies `a` and `b`',
            fr  = 'Multiplie `a` et `b`'
        )
        def multiply(a, b):
            return a * b
        
        help(multiply)  # displays the english docstring
        pring(get_prompt(multiply, 'fr')) # displays the French text
        ```
        
        Adds the `prompts` attribute to the wrapped function
    """
    def wrapper(fn):
        fn.prompts  = prompt
        fn.__doc__  = get_prompt(kwargs, 'en')
        return fn
    if prompt is None: prompt = Prompt(** kwargs)
    return wrapper
