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

import re

from functools import wraps

_newline_re = r'(?<![\'"])\n(?![\'"])'

def add_prompt_wrapper(task, *, fn = None):
    def wrapper(fn):
        @wraps(fn)
        def inner(self, * args, lang = None, prompts = None, ** kwargs):
            if not lang: lang = self.lang
            
            if isinstance(prompts, dict):
                kwargs = {** get_prompts(task, lang, prompts = prompts), ** kwargs}
            elif isinstance(prompts, str):
                kwargs = {** get_prompts(prompts, lang), ** kwargs}
            
            return fn(self, * args, lang = lang, **  {** get_prompts(task, lang), ** kwargs})
        
        inner.prompt_key = task
        return inner
    return wrapper if fn is None else wrapper(fn)

def get_prompts(task, lang = None, *, prompts = None):
    if prompts is None:
        global _prompts
        prompts = _prompts
    
    if task in prompts: prompts = prompts[task]
    
    result = {}
    for key, prompt in prompts.items():
        if isinstance(prompt, dict):
            if len(prompt) == 1: prompt = list(prompt.values())[0]
            else:   prompt = prompt[lang if lang in prompt else 'en']
        result[key] = prompt
    
    return result

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

def split_lines(text):
    is_string = False
    
    indexes = []
    for i, c in enumerate(text):
        if (c in ('"' "'")) and (i == 0 or i == len(text) - 1 or not text[i-1].isalnum() or not text[i+1].isalnum()):
            is_string = not is_string
        elif c == '\n' and not is_string:
            indexes.append(i)
        
    return [
        text[idx + 1 : indexes[i + 1]].strip('\n') for i, idx in enumerate(indexes[:-1])
    ]
        
def count_indent(line):
    for i, c in enumerate(line):
        if not c.isspace(): return i
    return len(line)
    
def dedent(text):
    lines   = split_lines(text)
    if not lines: return text
    common   = min(count_indent(l) for l in lines if l.strip())
    return '\n'.join([l.replace(' ' * common, '', 1) for l in lines]).strip()

_prompts_qa = {
    'personnality_prompt'   : {
        'fr'    : "Tu es un assistant IA. Tu dois répondre de manière brève et précise aux questions de l'utilisateur.",
        'en'    : "You are an AI assistant. You have to answer the best as possible to the user's queries"
    },
    'system_prompt' : {
        'fr'    : "{personnality_prompt}",
        'en'    : "{personnality_prompt}"
    },
    'format'    : {
        'fr'    : dedent("""
            {%- if paragraphs and not documents -%}
                {%- set documents = paragraphs -%}
            {%- endif -%}
            
            {%- if documents -%}
                {%- for doc in documents %}
                    {%- if not threshold is defined or 'score' not in doc or doc.score >= threshold %}
                        {% if 'filename' in doc and (loop.index == 1 or doc.filename != documents[loop.index - 2].filename) %}
                            {{- "<|start_header_id|>document<|end_header_id|>\n" }}
                            {{- "Fichier : " + doc.filename + "\n\n" }}
                        {% elif 'url' in doc and (loop.index == 1 or doc.url != documents[loop.index - 2].url) %}
                            {{- "<|start_header_id|>document<|end_header_id|>\n" }}
                            {{- "URL : " + doc.url + "\n\n" }}
                        {%- endif %}

                        {% if 'title' in doc and (loop.index == 1 or doc.title != documents[loop.index - 2].title) %}
                            {{- "Titre du document : " + doc.title + "\n" }}
                        {%- elif doc.section_titles %}
                            {{- "Titre des sections : " + ' > '.join(doc.section_titles) + "\n" }}
                        {%- endif %}
                        {{- doc.text.strip("\n") + "\n\n" }}
                    {%- endif %}
                {%- endfor %}
            {% endif %}

            {% if detailed %}
                {{- "Réponds de la manière la plus précise et détaillée possible !\n\n" -}}
            {% endif %}
            
            {%- if prefix -%}
                {{- prefix -}}
            {%- endif -%}
            
            {{- text -}}
            """),
        'en'    : dedent("""
            {%- if paragraphs and not documents -%}
                {%- set documents = paragraphs -%}
            {%- endif -%}
            
            {%- if documents -%}
                {%- for doc in documents %}
                    {%- if not threshold is defined or 'score' not in doc or doc.score >= threshold %}
                        {{- "<|start_header_id|>document<|end_header_id|>\n" }}
                        {% if 'filename' in doc and (loop.index == 1 or doc.filename != documents[loop.index - 2].filename) %}
                            {{- "File : " + doc.filename + "\n\n" }}
                        {% elif 'url' in doc and (loop.index == 1 or doc.url != documents[loop.index - 2].url) %}
                            {{- "URL : " + doc.url + "\n\n" }}
                        {%- endif %}

                        {% if 'title' in doc and (loop.index == 1 or doc.title != documents[loop.index - 2].title) %}
                            {{- "Title : " + doc.title + "\n" }}
                        {%- elif doc.section_titles %}
                            {{- "Section titles : " + ' > '.join(doc.section_titles) + "\n" }}
                        {%- endif %}
                        {{- doc.text + "\n\n" }}
                    {%- endif %}
                {%- endfor %}
            {% endif %}

            {% if detailed %}
                {{- "Answer in a very precise and detailed way !\n\n" -}}
            {% endif %}
            
            {%- if prefix -%}
                {{- prefix -}}
            {%- endif -%}
            
            {{- text -}}
            """)
    },
    'message_format'    : {
        'fr'    : dedent("""
        {% if user %}
            {% if message is undefined or message.role == 'user' %}
                {{- "Message de " + user + "\n\n" -}}
            {% endif %}
        {% endif %}

        {{- text -}}
        """),
        'en'    : dedent("""
        {% if user %}
            {% if message is undefined or message.role == 'user' %}
                {{- "Message from " + user + "\n\n" -}}
            {% endif %}
        {% endif %}

        {{- text -}}
        """)
    }
}

_prompts_rag    = {
    'format'    : {
        'fr'    : """{{- "\nVoici différentes sources d'informations :\n\n" }}\n""" + \
            _prompts_qa['format']['fr'] + "\n\n" + \
            dedent("""
            {{- "\n\nRéponds UNIQUEMENT sur base des informations fournies !\n" -}}

            {% if add_source %}
                {{- "La réponse doit explicitement mentionner les sources, par exemple en commençant par `comme mentionné dans ...`, ou `Selon ...`." -}}
            {% endif %}

            {{- "Si aucune information n'est pertinente, demande plus de précisions à l'utilisateur,\nSi la question n'a pas de rapport avec les informations, ne réponds pas.\n" -}}
            """),
        'en'    : """{{- "\nHere are some sources of information :\n\n" }}\n""" + \
            _prompts_qa['format']['en'] + "\n\n" + \
            dedent("""
            {{- "\n\nAnswer ONLY based on the provided information !\n" -}}

            {% if add_source %}
                {{- "The answer must explicitely mention the source of the information, e.g., with `based on ...` or `As mentionned in ...`" -}}
            {% endif %}

            {{- "If no information is relevant to the question, ask more information to the user.\nIf the question is irrelevant or not related to any topic from the provided information, do not answer.\n" -}}
            """)
    }
}


_prompts_translation    = {
    'prefix'    : {
        'fr'    : 'Traduis ce texte en anglais :\n\n',
        'en'    : 'Translate this text in French :\n\n'
    },
    'answer_start'  : {
        'fr'    : 'La traduction est :\n\n',
        'en'    : 'The translation is :\n\n'
    }
}

_prompts_reformulation  = {
    'prefix'    : {
        'fr'    : 'Réécris ce texte **sans expliquer les changements** en améliorant le style, la grammaire et la fluidité :\n\n',
        'en'    : 'Rewrite this text, without explaining your changes, by improving grammar, style and fluency :\n\n'
    }
}

_prompts_description  = {
    'prefix'    : {
        'fr'    : 'Décris de manière générale le contenu de ce texte :\n\n',
        'en'    : 'Describe the general content of this text :\n\n'
    }
}

_prompts_summarization  = {
    'prefix'    : {
        'fr'    : 'Ecris un résumé de ce texte en gardant les informations importantes :\n\n',
        'en'    : 'Write a short and concise summary of this text while keeping all the important information :\n\n'
    }
}

_prompts_entity_extraction  = {
    'format'    : {
        'fr'        : _prompts_qa['format']['fr'] + "\n\n" + dedent("""
            {{- "Retrouve les informations demandées et fais-en un résumé clair et concis au format JSON. Si une information est manquante, mets `null`.\n\n" }}

            {{- "Voici les informations à extraire : {\n" }}
            {%- for key, value in entities.items() %}
                {{- "    '" + key + "' : # " + value + "\n" }}
            {%- endfor %}
            {{- "}\n" }}
            """),
        'en'        : _prompts_qa['format']['en'] + "\n\n" + dedent("""
            {{- "Find the requested information in the below text, and add them in the JSON structure. If an information is missing, set the value `null`.\n\n" }}

            {{- "Here is the requested information : {\n" }}
            {%- for key, value in entities.items() %}
                {{- "    '" + key + "' : # " + value + "\n" }}
            {%- endfor %}
            {{- "}\n" }}
            """)
    },
    'answer_start'  : {
        'fr'    : "Voici les informations demandées :\n\n```json\n",
        'en'    : "Here is the requested information :\n\n```json\n"
    }
}

_prompts    = {
    'rag'   : _prompts_rag,
    'answer'    : _prompts_qa,
    'describe'  : _prompts_description,
    'summarize' : _prompts_summarization,
    'translate' : _prompts_translation,
    'reformulate'   : _prompts_reformulation,
    'extract_entities'  : _prompts_entity_extraction
}
