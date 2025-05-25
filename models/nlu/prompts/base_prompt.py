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

_newline_re = r'(?<![\'"])\n(?![\'"])'

class Prompt:
    def __init__(self, text = None, /, *, _args = (), _kwargs = {}, ** kwargs):
        if text: kwargs['en'] = text
        
        self._args  = _args
        self._kwargs    = _kwargs
        self.translations   = kwargs
    
    def __getitem__(self, lang):
        return get_translation(self.translations, lang).format(* self._args, ** self._kwargs)
    
    def __setitem__(self, lang, value):
        self.translations[lang] = value
    
    def __str__(self):
        return self['en']
    
    def __repr__(self):
        return self['en']
    
    def items(self):
        return self.translations.items()
    
    def update(self, translations):
        self.translations.update(translations)
    
    def format(self, * args, ** kwargs):
        return Prompt(_args = args, _kwargs = kwargs, ** self.translations)


def get_translation(translations, lang):
    if isinstance(translations, Prompt):    return translations[lang]
    elif isinstance(translations, str):     return translations
    elif len(translations) == 1:            return list(translations.values())[0]
    else:   return translations[lang if lang in translations else 'en']

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
    