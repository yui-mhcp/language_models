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
                {{- "\n\n## Conversation History" -}}
            {%- elif prompt_format == 'html' -%}
                {{- "\n\n<conversation>" -}}
            {%- endif -%}
            {{- "\n\nYou have access to the latest messages from the conversation. Use them to contextualize your response and reasoning.\n" -}}
        {%- endif -%}
    """),
    'personnality_format'   : dedent("""
        {%- if prompt_format == 'markdown' -%}
            {{- "## Personality\n\n" -}}
        {%- elif prompt_format == 'html' -%}
            {{- "<personality>\n" -}}
        {%- endif -%}

        {{- personnality -}}

        {%- if prompt_format == 'html' -%}
            {{- "\n</personality>" -}}
        {%- endif -%}
    """),
    'paragraphs_format' : dedent("""
        {%- if prompt_format == 'markdown' -%}
            {{- "## Information\n\n" -}}
        {%- elif prompt_format == 'html' -%}
            {{- "<information>\n" -}}
        {%- endif -%}

        {{- "You have access to this information to help you answer. **Use it if relevant**!" -}}
        
        {%- for para in paragraphs -%}
            {{- "\n\n" -}}
            {%- if loop.first -%}
                {%- set last_para = {} -%}
            {%- else -%}
                {%- set last_para = paragraphs[loop.index - 2] -%}
            {%- endif -%}
            
            {% if para['filename'] and para['filename'] != last_para['filename'] -%}
                {{- "- File: " + basename(para['filename']) + "\n" -}}
                {% if 'title' in para %}
                    {{- "- Title: " + para['title'] + "\n" -}}
                {%- endif -%}
            {% elif para['url'] and para['url'] != last_para['url'] -%}
                {{- "- URL: " + para['url'] + "\n" -}}
                {% if 'title' in para %}
                    {{- "- Title: " + para['title'] + "\n" -}}
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
            {{- "\n</information>\n" -}}
        {%- endif -%}
    """),
    'python_tools_format'   : dedent("""
        {%- if prompt_format == 'markdown' -%}
            {{- "\n\n## IPython Environment" -}}
        {%- elif prompt_format == 'html' -%}
            {{- "\n\n<ipython>" -}}
        {%- endif -%}

        {{- "\n\nYou are a Python programming expert, and you have access to a code interpreter. You can therefore write code to help you answer more complex questions, or questions requiring mathematical operations.\n" -}}
        {{- "Use the `print` function to display the results you want to access.\n" -}}

        {%- if python_tools -%}
            {{- "You have access to these functions allowing you to perform complex operations or search for additional information:\n```python" -}}
            {% for tool in python_tools %}
                {{- "\n" + tool.to_signature(lang = lang) + "\n" -}}
            {% endfor %}
            {{- "```\n\n" -}}
        {%- endif -%}

        {{- "To execute code, use the standard format:\n```python\n... # your python code\n```\n" -}}
        {{- "Your code will be executed in a restricted Python environment. You only have access to standard libraries (except `os`), as well as `numpy`.\n" -}}
        {{- "\nReason step by step:" -}}
        {{- "\n1) If no function is relevant, answer directly." -}}
        {{- "\n2) If you are missing information to call a function, ask the user for it." -}}
        {{- "\n3) Otherwise call the relevant function(s) in valid Python code." -}}
        {{- "\nTest your code by displaying results (with `print`), and correct yourself if the result is not as expected!" -}}

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
            {{- "\n\n## Pinned Messages" -}}
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
        "You are an AI assistant. You must respond to user requests as best as possible."
    )
}

prompts_expert   = {
    'personnality' : (
        "You are an expert in {domain} with the following background: {background}\n\n",
        "Answer the questions you are asked as best as possible."
    )
}

prompts_rag    = {
    'prefix'    : dedent("""
        {{- "\n\nAnswer ONLY based on the provided information!\n" -}}

        {% if add_source %}
            {{- "The answer must explicitly mention the sources, for example by starting with `as mentioned in ...`, or `According to ...`." -}}
        {% endif %}

        {{- "If no information is relevant, ask the user for more details.\nIf the question is not related to the information, do not answer.\n" -}}
    """)
}


prompts_translate    = {
    'prefix'    : dedent("""
        {%- if not target_lang -%}
            {% set target_lang = 'French' %}
        {%- endif -%}
        
        {{- "Translate this text into " + target_lang + ":\n\n" -}}
    """),
    'answer_start'  : 'The translation is:\n\n',
}

prompts_reformulate  = {
    'prefix'    : 'Rewrite this text **without explaining the changes** by improving the style, grammar, and fluency:\n\n'
}

prompts_describe  = {
    'prefix'    : 'Describe in general terms the content of this text:\n\n'
}

prompts_summarize  = {
    'prefix'    : 'Write a summary of this text keeping the important information:\n\n'
}

prompts_extract_entities  = {
    'format'    : dedent("""
        {{- text -}}

        {{- "\nBased on the provided text, find the requested information and make a clear and concise summary in JSON format. If any information is missing, put `null`.\n\n" }}

        {{- "Here is the information to extract:\n```json\n{\n" -}}
        {%- for key, value in entities.items() -%}
            {{- "    '" + key + "': # " + value + "\n" -}}
        {%- endfor -%}
        {{- "}\n```\n" -}}
        {{- "Answer using the same JSON format, and use the same entry names." -}}
    """),
    'answer_start'  : "Here is the requested information:\n\n```json\n",
    'stop_condition'    : lambda text: text.rstrip().endswith('```')
}