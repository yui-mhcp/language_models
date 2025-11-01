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

from .prompt import dedent

_default_formats    = {
    'system_prompt' : dedent("""
        {%- if personnality -%}
            {{- personnality + "\n\n" -}}
        {%- endif -%}

        {%- if python_tools -%}
            {{- python_tools + "\n\n" -}}
        {%- endif -%}
        
        {%- if paragraphs and not paragraphs_in_last_message -%}
            {{- paragraphs + "\n\n" -}}
        {%- endif -%}

        {%- if memories -%}
            {{- memories + "\n\n" -}}
        {%- endif -%}

        {%- if instructions and not instruct_in_last_message -%}
            {{- instructions + "\n\n" -}}
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
    'personnality_format'   : dedent("""
        {%- if prompt_format == 'markdown' -%}
            {{- "## Personnalité\n\n" -}}
        {%- elif prompt_format == 'html' -%}
            {{- "<personnalité>\n" -}}
        {%- endif -%}

        {{- personnality -}}

        {%- if prompt_format == 'html' -%}
            {{- "\n</personnalité>" -}}
        {%- endif -%}
    """),
    'paragraphs_format' : dedent("""
        {%- if prompt_format == 'markdown' -%}
            {{- "## Informations\n\n" -}}
        {%- elif prompt_format == 'html' -%}
            {{- "<informations>\n" -}}
        {%- endif -%}

        {{- "Tu as accès à ces informations pour t'aider à répondre. **Utilise-les si elles sont pertinentes** !" -}}
        
        {%- for para in paragraphs -%}
            {{- "\n\n" -}}
            {%- if loop.first -%}
                {%- set last_para = {} -%}
            {%- else -%}
                {%- set last_para = paragraphs[loop.index - 2] -%}
            {%- endif -%}
            
            {% if para['filename'] and para['filename'] != last_para['filename'] -%}
                {{- "- Fichier : " + basename(para['filename']) + "\n" -}}
                {% if 'title' in para %}
                    {{- "- Titre : " + para['title'] + "\n" -}}
                {%- endif -%}
            {% elif para['url'] and para['url'] != last_para['url'] -%}
                {{- "- URL : " + para['url'] + "\n" -}}
                {% if 'title' in para %}
                    {{- "- Titre : " + para['title'] + "\n" -}}
                {%- endif -%}
            {%- endif -%}

            {% if 'page' in para and para['page'] != last_para['page'] -%}
                {{- "- Page #" + para['page']|string + "\n" -}}
            {%- endif -%}
            
            {%- if 'type' not in para -%}
                {{- para['text']|trim -}}
            {%- elif para['type'] == 'text' -%}
                {{- para['text'] | trim -}}
            {%- elif para['type'] == 'code' -%}
                {{- "```" + para["language"] + "\n" + para["text"] | trim + "\n```" -}}
            {%- elif para['type'] == 'list' -%}
                {%- for item in para['items'] -%}
                    {{- "\n- " + item|string }}
                {%- endfor -%}
            {%- elif para['type'] == 'table' -%}
                {%- for item in para['rows'] -%}
                    {{- "\n- " + item|string }}
                {%- endfor -%}
            {%- endif -%}
        {%- endfor -%}

        {%- if prompt_format == 'html' -%}
            {{- "\n</informations>\n" -}}
        {%- endif -%}
    """),
    'python_tools_format'   : dedent("""
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
    """),
    'instructions_format'   : dedent("""
        {%- if prompt_format == 'markdown' -%}
            {{- "\n\n## Instructions" -}}
        {%- elif prompt_format == 'html' -%}
            {{- "\n\n<instructions>" -}}
        {%- endif -%}

        {% for instruct in instructions %}
            {{- "\n- " + instruct -}}
        {% endfor %}

        {%- if prompt_format == 'html' -%}
            {{- "\n</instructions>" -}}
        {%- endif -%}
    """),
    'pinned_format' : dedent("""
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

    """),
    'message_format'    : None,
    'last_message_format'   : dedent("""
        {%- if messages and messages|length > 2 and prompt_format == 'html'  -%}
            {{- "\n</conversation>\n\n" -}}
        {%- endif -%}
        
        {%- if paragraphs and paragraphs_in_last_message -%}
            {{- paragraphs + "\n\n" -}}
        {% endif %}

        {%- if pinned_messages -%}
            {{- pinned_messages + "\n\n" -}}
        {%- endif -%}
    
        {%- if instructions and instruct_in_last_message -%}
            {{- instructions + "\n\n" -}}
        {%- endif -%}

        {{- text -}}
    """)


}

prompts_default    = {
    ** _default_formats,

    'personnality'   : (
        "Tu es un assistant IA. Tu dois répondre du mieux possible aux demandes de l'utilisateur."
    )
}

prompts_expert   = {
    'personnality' : (
        "Tu es un expert en {domain} avec le background suivant : {background}\n\n",
        "Réponds du mieux possible aux questions qui te sont posées."
    )
}

prompts_rag    = {
    'prefix'    : dedent("""
        {{- "\n\nRéponds UNIQUEMENT sur base des informations fournies !\n" -}}

        {% if add_source %}
            {{- "La réponse doit explicitement mentionner les sources, par exemple en commençant par `comme mentionné dans ...`, ou `Selon ...`." -}}
        {% endif %}

        {{- "Si aucune information n'est pertinente, demande plus de précisions à l'utilisateur,\nSi la question n'a pas de rapport avec les informations, ne réponds pas.\n" -}}
    """)
}


prompts_translate    = {
    'prefix'    : dedent("""
        {%- if not target_lang -%}
            {% set target_lang = 'anglais' %}
        {%- endif -%}
        
        {{- "Traduis ce texte en " + target_lang + ":\n\n" -}}
    """),
    'answer_start'  : 'La traduction est :\n\n',
}

prompts_reformulate  = {
    'prefix'    : 'Réécris ce texte **sans expliquer les changements** en améliorant le style, la grammaire et la fluidité :\n\n'
}

prompts_describe  = {
    'prefix'    : 'Décris de manière générale le contenu de ce texte :\n\n'
}

prompts_summarize  = {
    'prefix'    : 'Ecris un résumé de ce texte en gardant les informations importantes :\n\n'
}

prompts_extract_entities  = {
    'format'    : dedent("""
        {{- text -}}
        
        {{- "Retrouve les informations demandées et fais-en un résumé clair et concis au format JSON. Si une information est manquante, mets `null`.\n\n" }}

        {{- "Voici les informations à extraire :\n```json\n{\n" }}
        {%- for key, value in entities.items() %}
            {{- "    '" + key + "': # " + value + "\n" }}
        {%- endfor %}
        {{- "}\n```\n" -}}
    """),
    'answer_start'  : "Voici les informations demandées :\n\n```json\n",
    'stop_condition'    : lambda text: text.rstrip().endswith('```')
}
