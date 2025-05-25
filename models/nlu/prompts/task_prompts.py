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
                {% set lang = 'en' -%}
            {%- endif -%}
            
            {{- "## Personnalité\n\n" -}}
            {{- personnality_prompt -}}
            
            {%- if is_vocal -%}
                {{- "\n\n## Style de réponse\n\n" -}}
                {{- "Tu es actuellement dans un appel vocal. Réponds de manière adaptée pour que ta réponse soit fluide et agréable à écouter. Privilégie les réponses courtes et directes.\n" -}}
                {{- "**ATTENTION** : les messages de l'utilisateur peuvent contenir des erreurs de transcription / d'orthographe. Essaye d'interpréter quand c'est possible, et sinon, demande de répéter." -}}
            {%- endif -%}

            {%- if python_tools -%}
                {{- "\n\n## Outils\n\n" -}}
                {{- "Tu es un expert en programmation python, ce qui te permet de répondre à des questions complexes en utilisant un interpréteur de code.\n" -}}
                {{- "Tu as accès à différents outils (fonctions python) que tu peux utiliser pour t'aider à répondre :\n```python" -}}
                {% for tool in python_tools %}
                    {{- "\n" + tool.to_signature(lang = lang) + "\n" -}}
                {% endfor %}
                {{- "```\n\n" -}}

                {{- "Pour utiliser ces outils, écris un code python :\n```python\n... # ton code python qui appelle un / plusieurs outil(s)\n```\n" -}}
                {{- "Ton code sera exécuté dans un environnement python restreint avec un accès direct aux fonctions définies ci-dessus (**ne les redéfinis pas**).\n" -}}
                {{- "\nRaisonne par étape :" -}}
                {{- "\n1) Si aucun outil n'est pertinent, réponds directement." -}}
                {{- "\n2) S'il te manque des informations pour appeler un outil, demande-les à l'utilisateur." -}}
                {{- "\n3) Sinon appelle le/les outil(s) dans un code python valide." -}}
                {{- "\n**Attention** : tu n'as pas accès au module `os`." -}}
                
                {%- if files -%}
                    {{- "\n\n## Fichiers\n" -}}
                    {{- "Tu as accès à ces fichiers pour t'aider à répondre :" -}}
                    {%- for file in files -%}
                        {{- "\n- " + file -}}
                    {%- endfor -%}
                    {{- "\n\nNe les lit pas manuellement ! Utilise les outils de lecture/recherche (comme `rag`) pour en extraire des informations." -}}
                {%- endif -%}
            {%- endif -%}
            
            {%- if documents -%}
                {{- "\n\n## Informations\n\n" -}}
                {{- "Tu as accès à ces informations pour t'aider à répondre. Utilise-les si nécessaire.\n" -}}
                {%- for doc in documents %}
                    {%- if not threshold is defined or 'score' not in doc or doc.score >= threshold %}
                        {% if 'filename' in doc and (loop.index == 1 or doc.filename != documents[loop.index - 2].filename) -%}
                            {{- "[CHEMIN] " + doc.filename + "\n" }}
                        {% elif 'url' in doc and (loop.index == 1 or doc.url != documents[loop.index - 2].url) %}
                            {{- "[URL] " + doc.url + "\n" }}
                        {%- endif -%}

                        {% if 'title' in doc %}
                            {{- "[TITRE] " + doc.title + "\n" }}
                        {%- endif -%}
                        {%- if doc.section_titles %}
                            {{- "[SECTION] " + ' > '.join(doc.section_titles) + "\n" }}
                        {%- endif -%}
                        {%- if doc.pages -%}
                            {{- "[PAGE] " + doc.pages|string + "\n" -}}
                        {%- endif -%}
                        
                        {{- doc.text.strip("\n") + "\n\n" }}
                    {%- endif -%}
                {%- endfor -%}
            {% endif %}

            {%- if messages and messages|length > 2  -%}
                {{- "\n\n## Historique de conversation\n\n" -}}
                {{- "Tu as accès aux derniers messages de la conversation. Utilise-les pour contextualiser ta réponse et tes raisonnements.\n" -}}
            {%- endif -%}
        """),
        'en'    : dedent("""
            {{- personnality_prompt -}}
            
            {%- if is_vocal -%}
                {{- "\n\nYou are currently in a voice call. Make sure to provide fluent answers adapted to vocal discussions." -}}
            {%- endif -%}
        """),
    },
    'last_message_format'   : {
        'fr'    : dedent("""
        {%- if messages and messages|length > 2  -%}
            {{- "\n[FIN DE L'HISTORIQUE]\n\n" -}}
        {%- endif -%}
        
        {%- if pinned_messages -%}
            {{- "[MESSAGES EPINGLES]\n" -}}
            {%- for message in pinned_messages -%}
                {{- "[" + loop.index|string + "] " + message.content + "\n" -}}
            {%- endfor -%}
            {{- "\n[FIN DES MESSAGES EPINGLES]\n\n" -}}
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
    'format'    : {
        'fr'    : dedent("""
            {%- if detailed -%}
                {{- "Réponds de la manière la plus précise et détaillée possible !\n\n" -}}
            {%- endif -%}
            
            {%- if prefix -%}
                {{- prefix -}}
            {%- endif -%}
            
            {%- if is_vocal -%}
                {{- "Transcription :\n" -}}
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
        {%- if message.role == 'user' and (message.user or message.time) -%}
            {{- "[INFORMATIONS SUR LE MESSAGE]" -}}
            {%- if message.user -%}
                {{- "\n[NOM DE L'ENVOYEUR] " + message.user -}}
            {%- endif -%}
            {%- if message.time -%}
                {{- "\n[DATE] " + timestamp_to_str(message.time) -}}
            {%- endif -%}
            {{- "\n\n" -}}
        {%- endif -%}
        
        {{- text -}}
        """),
        'en'    : '{text}'
    }
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
        'fr'    : """{{- "\nVoici différentes sources d'informations :\n\n" }}\n""" + \
            prompts_default['format']['fr'] + "\n\n" + \
            dedent("""
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
