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

from .templates import get_translation, get_template, set_template, add_template_wrapper
from .prompt_formatter import PromptFormatter

def add_multilingual_docstring(prompt = None, /, ** kwargs):
    """
        Setup a multilingual docstring
        This is especially useful for tools description if the model should be multilingual

        Example :
        ```
        @add_multilingual_docstring(
            en  = 'Multiplies `a` and `b`',
            fr  = 'Multiplie `a` et `b`'
        )
        def multiply(a, b):
            return a * b

        help(multiply)  # displays the english docstring
        print(get_translation(multiply, 'fr')) # displays the French text
        ```

        Adds the `prompts` attribute to the wrapped function
    """
    def wrapper(fn):
        fn.prompts  = prompt
        fn.__doc__  = get_translation(kwargs, 'en')
        return fn
    if prompt is None: prompt = kwargs
    return wrapper
