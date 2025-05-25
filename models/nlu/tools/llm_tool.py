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

from .tool import Tool
from ..prompts import prompt_docstring

@prompt_docstring(
    en = "Use a LLM to answer a question",
    fr = "Utilise un LLM pour répondre à une question"
)
def ask(question : str, *, model, ** _):
    return model.answer(question)

@prompt_docstring(
    en = "Ask a `question` to an expert. Provide the expected expert background in the `background` argument",
    fr = "Pose une question à un expert. Spécifie le type d'expert recherché dans l'argument `background`"
)
def ask_expert(question : str, domain : str, background : str, *, model, **_):
    return model.ask_expert(question, domain = domain, background = background)

@prompt_docstring(
    en = "Translate `text` from `lang` to `target_lang`",
    fr = "Traduit la requête vers `target_lang`"
)
def translate(text : str, lang : str, target_lang : str, *, model, ** _):
    return model.translate(text, lang = lang, target_lang = target_lang)
    
@prompt_docstring(
    en = "Use a LLM to summarize `text`",
    fr = "Utilise un LLM pour résumer un texte"
)
def summarize(text : str, *, model, ** _):
    return model.summarize(text, lang = lang, target_lang = target_lang)


QATool          = Tool.from_function(ask)
ExpertTool      = Tool.from_function(ask_expert)
TranslationTool = Tool.from_function(translate)
SummarizationTool   = Tool.from_function(summarize)

LLMTools    = [v for k, v in globals().items() if isinstance(v, type) and issubclass(v, Tool)]
