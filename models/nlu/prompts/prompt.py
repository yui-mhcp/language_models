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

from utils.text import format_text

class Prompt:
    """
        This class represents an abstract prompt with possible translations
        
        Examples :
        ```
        prompt = Prompt(
            en  = 'Hello World !',
            fr  = 'Bonjour le monde !'
        )
        print(prompt['en'])                     # Hello World !
        print(get_translation(prompt, 'fr'))    # Bonjour le monde !
        ```
        
        It can also be formatted (like strings) :
        ```
        prompt = Prompt(
            en  = 'Hello {username} ! How are you today ?',
            fr  = "Bonjour {username} ! Comment allez-vous ajourd'hui ?"
        )
        # These two are equivalent
        print(prompt['en'].format(username = "Yui"))
        print(prompt.format(username = "Yui")['en'])
        ```
    """
    def __init__(self, _text = None, /, *, _args = (), _kwargs = {}, ** kwargs):
        if _text: kwargs['en'] = _text
        
        self._args  = _args
        self._kwargs    = _kwargs
        self.translations   = kwargs
    
    def __str__(self):
        return self['en']

    def __repr__(self):
        return '<Prompt lang={}>'.format(tuple(self.translations.keys()))
    
    def __len__(self):
        return len(self.translations)
    
    def __contains__(self, lang):
        return lang in self.translations
    
    def __getitem__(self, lang):
        return get_prompt(self.translations, lang, * self._args, ** self._kwargs)
    
    def __setitem__(self, lang, value):
        self.translations[lang] = value
    
    def items(self):
        return self.translations.items()
    
    def update(self, translations):
        self.translations.update(translations)
    
    def format(self, * args, ** kwargs):
        return Prompt(_args = args, _kwargs = kwargs, ** self.translations)


def get_prompt(prompts, /, lang, * args, ** kwargs):
    """
        Return the prompt for the given language (if available, fallback to 'en')
        if `args` or `kwargs` is provided, they are forwarded to `utils.text.format_text`
    """
    if hasattr(prompts, 'prompts'): prompts = prompts.prompts
    
    if isinstance(prompts, str):
        prompt = prompts
    elif isinstance(prompts, Prompt):
        prompt = prompts[lang]
    elif not isinstance(prompts, dict):
        raise ValueError('`prompt` should be a `Prompt`, `str` or `dict`, got {}'.format(prompts))
    elif len(prompts) == 1:
        prompt = list(prompts.values())[0]
    else:
        prompt = prompts[lang if lang in prompts else 'en']
    
    if args or kwargs:
        prompt = format_text(prompt, * args, ** kwargs)
    
    return prompt

def dedent(text):
    lines   = _split_lines(text)
    if not lines: return text
    common   = min(_count_indent(l) for l in lines if l.strip())
    return '\n'.join([l.replace(' ' * common, '', 1) for l in lines]).strip()

def _split_lines(text):
    is_string = False
    
    indexes = []
    for i, c in enumerate(text):
        if (c == '"') and (i == 0 or i == len(text) - 1 or not text[i-1].isalnum() or not text[i+1].isalnum()):
            is_string = not is_string
        elif c == '\n' and not is_string:
            indexes.append(i)
    
    if indexes[-1] != len(text): indexes.append(len(text))
    return [
        text[idx + 1 : indexes[i + 1]].strip('\n') for i, idx in enumerate(indexes[:-1])
    ]
        
def _count_indent(line):
    for i, c in enumerate(line):
        if not c.isspace(): return i
    return len(line)
    