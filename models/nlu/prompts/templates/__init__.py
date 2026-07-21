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

from utils.text import format_text

def get_template(task, lang = 'en', *, templates = None):
    """
        Return a mapping `{prompt_key : text}` associated to the given `task` and `lang`

        Arguments :
            - task  : a task representing a valid entry in `_templates`
            - lang  : the prompt language (if multiple translations are available)
            - templates : a nested mapping `{task : {prompt_key : {lang : text}}}`
                          If not provided, default to `_templates`
        Return :
            - templates : a mapping `{prompt_key : text}` for the given task / lang
                          if `task` is not in `templates`, return an empty `dict`
    """
    if templates is None:
        global _templates
        templates = _templates

    return {k : v[lang] for k, v in templates.get(task, {}).items() if lang in v}

def set_template(template, task = None, key = None, lang = None):
    """ Set a new template for the given task, prompt key and lang """
    if not task:
        for task, p in template.items():
            set_template(p, task, key, lang)
    elif key is None:
        for key, p in template.items():
            set_template(p, task, key, lang)
    elif lang is None:
        for lang, p in template.items():
            set_template(p, task, key, lang)
    else:
        _templates.setdefault(task, {}).setdefault(key, {})[lang] = template

def get_translation(value, /, lang = 'en', * args, ** kwargs):
    """
        Resolve a multilingual `value` to its text for the given `lang` (fallback to 'en')

        `value` may be :
            - a `str`   : returned as-is
            - a `dict`  : `{lang : text}` (a single-entry dict returns its only value)
            - any object exposing a `prompts` attribute (e.g. a function decorated with
              `add_multilingual_docstring`), in which case `value.prompts` is resolved instead

        if `args` or `kwargs` is provided, they are forwarded to `utils.text.format_text`
    """
    if hasattr(value, 'prompts'): value = value.prompts

    if isinstance(value, str):
        text = value
    elif not isinstance(value, dict):
        raise ValueError('`value` should be a `str` or `dict`, got {}'.format(value))
    elif len(value) == 1:
        text = list(value.values())[0]
    else:
        text = value[lang if lang in value else 'en']

    if args or kwargs:
        text = format_text(text, * args, ** kwargs)

    return text

def add_template_wrapper(_task, /, *, fn = None):
    """
        Wraps a function to automatically add all `task`-related templates to `kwargs`

        Example :
        ```
        @add_template_wrapper('translate')
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
                    if t: config.update(get_template(t, lang))

                kwargs = {** config, ** kwargs}

            return fn(self, * args, lang = lang, ** kwargs)

        inner.prompt_task = _task
        return inner

    return wrapper if fn is None else wrapper(fn)


_templates = {}
for _file in glob.glob(os.path.join(os.path.dirname(__file__), '*.py')):
    _lang = os.path.splitext(os.path.basename(_file))[0]
    if _lang.startswith('_'):
        continue

    _module = importlib.import_module('{}.{}'.format(__package__, _lang))
    for _k, _v in vars(_module).items():
        if _k.startswith('prompts_'):
            _task = _k[8:]
            for _prompt_key, _text in _v.items():
                set_template(_text, _task, _prompt_key, _lang)
