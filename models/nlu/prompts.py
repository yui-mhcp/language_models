# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps

def add_prompt_wrapper(task, *, fn = None):
    def wrapper(fn):
        @wraps(fn)
        def inner(self, * args, lang = None, ** kwargs):
            if lang is None: lang = self.lang
            
            for key, prompt in _prompts[task].items():
                if key not in kwargs:
                    if isinstance(prompt, dict):
                        prompt = prompt[lang if lang in prompt else 'en']
                    kwargs[key] = prompt
            
            return fn(self, * args, lang = lang, ** kwargs)
        return inner
    return wrapper if fn is None else wrapper(fn)

_prompts_qa = {
    'system_prompt' : {
        'fr'    : """
Tu es un assistant IA. Tu dois répondre de manière brève et précise aux questions de l'utilisateur.
Si des informations te sont fournies, utilise ces informations pour répondre **si** elles sont pertinentes.
""".strip(),
        'en'    : """
You are an AI assistant. You have to answer the best as possible to the user's queries.
If additional information are provided, you have to use them **if** they are relevant to answer the query.
""".strip(),
    }
}

_prompts_rag    = {
    'system_prompt' : {
        'fr'    : """
{{ "Tu es un assistant IA et tu dois répondre de manière brève et précise aux questions de l'utilisateur.\n" -}}

{{- "\nVoici différentes sources d'informations :\n\n" }}

{%- for doc in documents %}
    {%- if not threshold is defined or doc.score >= threshold %}
        {{- "<|start_header_id|>document<|end_header_id|>\n" }}
        {%- if 'filename' in doc %}
            {{- "Fichier : " + doc.filename + "\n\n" }}
        {%- elif 'url' in doc %}
            {{- "URL : " + doc.url + "\n\n" }}
        {%- endif %}
        
        {%- if 'title' in doc %}
            {{- "Titre du document : " + doc.title + "\n" }}
        {%- elif doc.section_titles %}
            {{- "Titre des sections : " + ' > '.join(doc.section_titles) + "\n" }}
        {%- endif %}
        {{- doc.text + "\n\n" }}
    {%- endif %}
{%- endfor %}
""".strip(),
        'en'    : """
{{- "You are an AI assistant, and you have to answer the best as possible to the user's queries.\n\n" -}}

{{- "\nHere are the provided information :\n\n" }}

{%- for doc in documents %}
    {%- if not threshold is defined or doc.score >= threshold %}
        {{- "<|start_header_id|>information<|end_header_id|>\n" }}
        {%- if 'filename' in doc %}
            {{- "Filename : " + doc.filename + "\n\n" }}
        {%- elif 'url' in doc %}
            {{- "URL : " + doc.url + "\n\n" }}
        {%- endif %}

        {%- if 'title' in doc %}
            {{- "Document title : " + doc.title + "\n" }}
        {%- elif doc.section_titles %}
            {{- "Section titles : " + ' > '.join(doc.section_titles) + "\n" }}
        {%- endif %}
        {{- doc.text + "\n\n" }}
    {%- endif %}
{%- endfor %}
""".strip()
    },
    'format'    : {
        'fr'    : """
{{- text -}}

{{- "\n\nRéponds UNIQUEMENT sur base des informations fournies !\n" -}}

{% if add_source %}
    {{- "La réponse doit explicitement mentionner les sources, par exemple en commençant par `comme mentionné dans ...`, ou `Selon ...`." -}}
{% endif %}

{{- "Si aucune information n'est pertinente, demande plus de précisions à l'utilisateur,\nSi la question n'a pas de rapport avec les informations, ne réponds pas.\n" -}}
""".strip(),
        'en'    : """
{{- text -}}

{{- "\n\nAnswer ONLY based on the provided information !\n" -}}

{% if add_source %}
    {{- "The answer must explicitely mention the source of the information, e.g., with `based on ...` or `As mentionned in ...`" -}}
{% endif %}

{{- "If no information is relevant to the question, ask more information to the user.\nIf the question is irrelevant or not related to any topic from the provided information, do not answer.\n" -}}
""".strip()
    }
}


_prompts_translation    = {
    'format'        : {
        'fr'    : 'Traduis ce texte en anglais :\n\n{text}',
        'en'    : 'Translate this text in French :\n\n{text}'
    }
}

_prompts_reformulation  = {
    'format'        : {
        'fr'    : 'Réécris ce texte **sans expliquer les changements** en améliorant le style, la grammaire et la fluidité :\n\n{text}',
        'en'    : 'Rewrite this text, without explaining your changes, by improving grammar, style and fluency :\n\n{text}'
    }
}

_prompts_summarization  = {
    'format'        : {
        'fr'    : 'Ecris un résumé de ce texte en gardant les informations importantes :\n\n{text}',
        'en'    : 'Write a short and concise summary of this text while keeping all the important information :\n\n{text}'
    }
}

_prompts_entity_extraction  = {
    'format'    : {
        'fr'        : """
{{- "Retrouve les informations demandées et fais-en un résumé clair et concis au format JSON. Si une information est manquante, mets `null`.\n\n" }}

{{- text + "\n\n" }}

{{- "Voici les informations à extraire : {\n" }}
{%- for key, value in entities.items() %}
    {{- "    '" + key + "' : # " + value + "\n" }}
{%- endfor %}
{{- "}\n" }}
""",
        'en'        : """
{{- "Find the requested information in the below text, and add them in the JSON structure. If an information is missing, set the value `null`.\n\n" }}

{{- text + "\n\n" }}

{{- "Here is the requested information : {\n" }}
{%- for key, value in entities.items() %}
    {{- "    '" + key + "' : # " + value + "\n" }}
{%- endfor %}
{{- "}\n" }}
"""
    }
}

_prompts    = {
    'qa'    : _prompts_qa,
    'rag'   : _prompts_rag,
    'translation'   : _prompts_translation,
    'reformulation' : _prompts_reformulation,
    'summarization' : _prompts_summarization,
    'entity_extraction' : _prompts_entity_extraction
}
