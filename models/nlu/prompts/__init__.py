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

from .task_prompts import *
from .base_prompt import Prompt, get_translation

_prompts = {}
for file in glob.glob(os.path.join(* __package__.split('.'), 'prompts_*.py')):
    module = importlib.import_module(file.replace(os.path.sep, '.')[:-3])
    
    lang = file[:-3].split('_')[-1]
    for k, v in vars(module).items():
        if k.startswith('prompts_'):
            task = k[8:]
            for ki, vi in v.items():
                _prompts.setdefault(task, {}).setdefault(ki, {}).update({lang : vi})


def add_prompt_wrapper(_task, /, *, fn = None):
    def wrapper(fn):
        @wraps(fn)
        def inner(self,
                  * args,
                  
                  task  = None,
                  lang  = None,
                  prompts   = None,
                  
                  _add_prompts  = True,
                  
                  ** kwargs
                 ):
            if _add_prompts:
                if not lang: lang = self.lang
                
                config = {}
                for t in (_task, task):
                    if not t: continue
                    config.update(get_prompts(t, lang))

                    if isinstance(prompts, dict):
                        config.update(get_prompts(t, lang, prompts = prompts))
                    elif isinstance(prompts, str):
                        config.update(get_prompts(prompts, lang))
                
                kwargs = {** config, ** kwargs}
            
            return fn(self, * args, lang = lang, ** kwargs)
        
        inner.prompt_key = _task
        return inner
    
    return wrapper if fn is None else wrapper(fn)

def prompt_docstring(prompt = None, /, ** kwargs):
    def wrapper(fn):
        fn.prompts  = prompt
        fn.__doc__  = get_translation(kwargs, 'en')
        return fn
    if prompt is None: prompt = Prompt(** kwargs)
    return wrapper

def get_prompts(task, lang = None, *, prompts = None):
    if prompts is None:
        global _prompts
        prompts = _prompts
        
        if task not in prompts: return {}
        prompts = prompts[task]
    
    return {k : get_translation(v, lang) for k, v in prompts.items()}

def set_prompt(prompt, task = None, key = None, lang = None):
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
        _prompts.setdefault(task, {}).setdefault(key, {}).update({lang : prompt})
