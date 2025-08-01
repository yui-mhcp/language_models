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

from .base_prompt import dedent

prompts_default    = {
    'personnality_prompt'   : {
        'fr'    : "Tu es un assistant IA. Tu dois répondre du mieux possible aux demandes de l'utilisateur.",
        'en'    : "You are an AI assistant. You have to answer the best as possible to the user's queries"
    },
    'system_prompt' : {
        'fr'    : dedent("""
            {%- if not lang -%}
                {% set lang = 'fr' -%}
            {%- endif -%}

            {%- if prompt_format == 'markdown' -%}
                {{- "## Personnalité\n\n" -}}
            {%- elif prompt_format == 'html' -%}
                {{- "<personnalité>\n" -}}
            {%- endif -%}
            
            {{- personnality_prompt -}}
            
            {%- if prompt_format == 'html' -%}
                {{- "\n</personnalité>" -}}
            {%- endif -%}

            {%- if is_vocal -%}
                {%- if prompt_format == 'markdown' -%}
                    {{- "\n\n## Style de réponse" -}}
                {%- elif prompt_format == 'html' -%}
                    {{- "\n\n<style>" -}}
                {%- endif -%}

                {{- "\n\nTu es actuellement dans un appel vocal. Réponds de manière adaptée pour que ta réponse soit fluide et agréable à écouter. Privilégie les réponses courtes et directes.\n" -}}
                {{- "**ATTENTION** : les messages de l'utilisateur peuvent contenir des erreurs de transcription / d'orthographe. Essaye d'interpréter quand c'est possible, et sinon, demande de répéter." -}}
                
                {%- if prompt_format == 'html' -%}
                    {{- "\n</style>" -}}
                {%- endif -%}
            {%- endif -%}

            {%- if python_tools or allow_code_execution -%}
                {%- if prompt_format == 'markdown' -%}
                    {{- "\n\n## Environnement ipython" -}}
                {%- elif prompt_format == 'html' -%}
                    {{- "\n\n<ipython>" -}}
                {%- endif -%}

                {{- "\n\nTu es un expert en programmation python, et tu as accès à un interpréteur de code. Tu peux donc écrire du code pour t'aider à répondre à des questions plus complexes, ou nécessitant des opérations mathématiques.\n" -}}
                {{- "Utilise la fonction `print` pour afficher les résultats auxquels tu veux avoir accès.\n" -}}
                
                {%- if python_tools -%}
                    {{- "Tu as accès à ces fonctions te permettant d'effectuer des opérations complexes, ou de rechercher des informations supplémentaires :\n```python" -}}
                    {% for tool in python_tools %}
                        {{- "\n" + tool.to_signature(lang = lang) + "\n" -}}
                    {% endfor %}
                    {{- "```\n\n" -}}
                {%- endif -%}

                {{- "Pour exécuter du code, utilise le format standard :\n```python\n... # ton code python\n```\n" -}}
                {{- "Ton code sera exécuté dans un environnement python restreint. Tu n'as accès qu'aux librairies standards (excepté `os`), ainsi qu'à `numpy`.\n" -}}
                {{- "\nRaisonne par étape :" -}}
                {{- "\n1) Si aucune fonction n'est pertinente, réponds directement." -}}
                {{- "\n2) Si il te manque des informations pour appeler une fonction, demande-les à l'utilisateur." -}}
                {{- "\n3) Sinon appelle la/les fonction(s) dans un code python valide." -}}
                {{- "\nTeste ton code en affichant des résultats (avec `print`), et corrige-toi si le résultat n'est pas celui attendu !" -}}
                
                {%- if prompt_format == 'html' -%}
                    {{- "\n</ipython>" -}}
                {%- endif -%}
            {%- endif -%}
            
            {%- if paragraphs -%}
                {%- if prompt_format == 'markdown' -%}
                    {{- "\n\n## Informations\n\n" -}}
                {%- elif prompt_format == 'html' -%}
                    {{- "\n<informations>\n" -}}
                {%- endif -%}
                
                {{- "Tu as accès à ces informations pour t'aider à répondre. **Utilise-les si elles sont pertinentes** !" -}}
                {%- for para in paragraphs %}
                    {{- "\n\n" -}}
                    {% if 'filename' in para and (loop.index == 1 or para.filename != paragraphs[loop.index - 2].filename) -%}
                        {{- "- Fichier : " + para.filename + "\n" -}}
                        {% if 'title' in para %}
                            {{- "- Titre : " + para.title + "\n" -}}
                        {%- endif -%}
                    {% elif 'url' in para and (loop.index == 1 or para.url != paragraphs[loop.index - 2].url) %}
                        {{- "- URL : " + para.url + "\n" -}}
                        {% if 'title' in para %}
                            {{- "- Titre : " + para.title + "\n" -}}
                        {%- endif -%}
                    {%- endif -%}

                    {%- if 'page' in para and (loop.index == 1 or para.page != paragraphs[loop.index - 2].page)  -%}
                        {{- "- Page #" + para.page|string + "\n" -}}
                    {%- endif -%}
                    {%- if 'section' in para and para.section %}
                        {{- "- Section : " + para.section[-1] + "\n" }}
                    {%- endif -%}
                    
                    {%- if para.type == 'code' -%}
                        {{- "```" + para.language or "text" + "\n" + para.text + "\n```" -}}
                    {%- elif para.type == 'list' -%}
                        {%- for item in para['items'] -%}
                            {{- "\n- " + item|string }}
                        {%- endfor -%}
                    {%- elif para.type == 'table' -%}
                        {%- for row in para.rows -%}
                            {{- "\n- " + row|string }}
                        {%- endfor -%}
                    {%- elif 'text' in para -%}
                        {{- para.text.strip("\n") -}}
                    {%- endif -%}
                {%- endfor -%}
            {% endif %}

            {%- if instructions and not instruct_in_last_message -%}
                {%- if prompt_format == 'markdown' -%}
                    {{- "\n\n## Instructions" -}}
                {%- elif prompt_format == 'html' -%}
                    {{- "\n\n<instructions>" -}}
                {%- endif -%}
                
                {% for instruct in instructions %}
                    {{- "\n- " + instruct -}}
                {% endfor %}
                {%- if python_tools -%}
                    {% for tool in python_tools %}
                        {% for instruct in tool.get_instructions(lang = lang) %}
                            {{- "\n- " + instruct -}}
                        {% endfor %}
                    {% endfor %}
                {%- endif -%}
                
                {%- if prompt_format == 'html' -%}
                    {{- "\n</instructions>" -}}
                {%- endif -%}
            {%- endif -%}
            
            {%- if messages and messages|length > 2  -%}
                {%- if prompt_format == 'markdown' -%}
                    {{- "\n\n## Historique de conversation" -}}
                {%- elif prompt_format == 'html' -%}
                    {{- "\n\n<conversation>" -}}
                {%- endif -%}
                {{- "\n\nTu as accès aux derniers messages de la conversation. Utilise-les pour contextualiser ta réponse et tes raisonnements.\n" -}}
            {%- endif -%}
        """),
        'en'    : dedent("""
            {{- personnality_prompt -}}
            
            {%- if is_vocal -%}
                {{- "\n\nYou are currently in a voice call. Make sure to provide fluent answers adapted to vocal discussions." -}}
            {%- endif -%}
        """),
    },
    'format'    : {
        'fr'    : dedent("""
            {%- if prefix -%}
                {{- prefix -}}
            {%- endif -%}
            
            {%- if is_vocal -%}
                {{- "Transcription :\n" -}}
            {%- endif -%}
            
            {{- text -}}
            """),
        'en'    : dedent("""
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
        {%- if message.role == 'user' and (message.user or message.time) -%}
            {%- if message.user -%}
                {{- "\n- Nom de l'utilisateur : " + message.user -}}
            {%- endif -%}
            {%- if message.time -%}
                {{- "\n- Date : " + timestamp_to_str(message.time) -}}
            {%- endif -%}
            {{- "\n\n" -}}
        {%- endif -%}
        
        {{- text -}}
        """),
        'en'    : '{text}'
    },
    'last_message_format'   : {
        'fr'    : dedent("""
        {%- if messages and messages|length > 2 and prompt_format == 'html'  -%}
            {{- "\n</conversation>\n\n" -}}
        {%- endif -%}
        
        {%- if pinned_messages -%}
            {%- if prompt_format == 'markdown' -%}
                {{- "\n\n## Messages épinglés" -}}
            {%- elif prompt_format == 'html' -%}
                {{- "\n\n<pinned>\n" -}}
            {%- endif -%}
            
            {%- for message in pinned_messages -%}
                {{- "[" + loop.index|string + "] " + message.content + "\n" -}}
            {%- endfor -%}
            {%- if prompt_format == 'html' -%}
                {{- "\n</pinned>" -}}
            {%- endif -%}
        {%- endif -%}
    
        {%- if instructions and instruct_in_last_message -%}
            {%- if prompt_format == 'markdown' -%}
                {{- "\n\n## Instructions" -}}
            {%- elif prompt_format == 'html' -%}
                {{- "\n\n<instructions>" -}}
            {%- endif -%}

            {% for instruct in instructions %}
                {{- "\n- " + instruct -}}
            {% endfor %}
            {%- if python_tools -%}
                {% for tool in python_tools %}
                    {% for instruct in tool.get_instructions(lang = lang) %}
                        {{- "\n- " + instruct -}}
                    {% endfor %}
                {% endfor %}
            {%- endif -%}

            {%- if prompt_format == 'html' -%}
                {{- "\n</instructions>" -}}
            {%- endif -%}
        {%- endif -%}

        {{- text -}}
        """),
        'en'    : dedent("""
        {%- if pinned_messages -%}
            {{- "\n[PINNED MESSAGES]\n" -}}
            {%- for message in pinned_messages -%}
                {{- "[" + loop.index|string + "] " + message.content + "\n" -}}
            {%- endfor -%}
            {{- "\n" -}}
        {%- endif -%}
    
        {{- text -}}
        """),
    },
}


prompts_qa = {}

prompts_expert   = {
    'personnality_prompt' : {
        'fr'    : "Tu es un expert en {domain} avec le background suivant : {background}\n\nRéponds du mieux possible aux questions.",
        'en'    : "You are an expert in {domain} with the following background : {background}\n\nAnswer the best as possible to the user queries."
    }
}

prompts_rag    = {
    'format'    : {
        'fr'    : dedent("""
            {{- "\n\nRéponds UNIQUEMENT sur base des informations fournies !\n" -}}

            {% if add_source %}
                {{- "La réponse doit explicitement mentionner les sources, par exemple en commençant par `comme mentionné dans ...`, ou `Selon ...`." -}}
            {% endif %}

            {{- "Si aucune information n'est pertinente, demande plus de précisions à l'utilisateur,\nSi la question n'a pas de rapport avec les informations, ne réponds pas.\n" -}}
            """),
        'en'    : """{{- "\nHere are some sources of information :\n\n" }}\n""" + \
            prompts_default['format']['en'] + "\n\n" + \
            dedent("""
            {{- "\n\nAnswer ONLY based on the provided information !\n" -}}

            {% if add_source %}
                {{- "The answer must explicitely mention the source of the information, e.g., with `based on ...` or `As mentionned in ...`" -}}
            {% endif %}

            {{- "If no information is relevant to the question, ask more information to the user.\nIf the question is irrelevant or not related to any topic from the provided information, do not answer.\n" -}}
            """)
    }
}


prompts_translation    = {
    'prefix'    : {
        'fr'    : dedent("""
        {%- if not target_lang -%}
            {% set target_lang = 'anglais' %}
        {%- endif -%}
        
        {{- "Traduis ce texte en " + target_lang + "\n\n" -}}
        """),
        'en'    : 'Translate this text in French :\n\n'
    },
    'answer_start'  : {
        'fr'    : 'La traduction est :\n\n',
        'en'    : 'The translation is :\n\n'
    }
}

prompts_reformulation  = {
    'prefix'    : {
        'fr'    : 'Réécris ce texte **sans expliquer les changements** en améliorant le style, la grammaire et la fluidité :\n\n',
        'en'    : 'Rewrite this text, without explaining your changes, by improving grammar, style and fluency :\n\n'
    }
}

prompts_description  = {
    'prefix'    : {
        'fr'    : 'Décris de manière générale le contenu de ce texte :\n\n',
        'en'    : 'Describe the general content of this text :\n\n'
    }
}

prompts_summarization  = {
    'prefix'    : {
        'fr'    : 'Ecris un résumé de ce texte en gardant les informations importantes :\n\n',
        'en'    : 'Write a short and concise summary of this text while keeping all the important information :\n\n'
    }
}

prompts_entity_extraction  = {
    'format'    : {
        'fr'        : prompts_default['format']['fr'] + "\n\n" + dedent("""
            {{- "Retrouve les informations demandées et fais-en un résumé clair et concis au format JSON. Si une information est manquante, mets `null`.\n\n" }}

            {{- "Voici les informations à extraire :\n```json\n{\n" }}
            {%- for key, value in entities.items() %}
                {{- "    \"" + key + "\": # " + value + "\n" }}
            {%- endfor %}
            {{- "}\n```\n" -}}
            """),
        'en'        : prompts_default['format']['en'] + "\n\n" + dedent("""
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
    },
    'stop_words'    : '```'
}
