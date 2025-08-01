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
    'personnality_prompt'   : (
        "You are an AI assistant. You must respond as best as possible to user requests."
    ),
    'system_prompt' : dedent("""
        {%- if not lang -%}
            {% set lang = 'en' -%}
        {%- endif -%}

        {%- if prompt_format == 'markdown' -%}
            {{- "## Personality\n\n" -}}
        {%- elif prompt_format == 'html' -%}
            {{- "<personality>\n" -}}
        {%- endif -%}

        {{- personnality_prompt -}}

        {%- if prompt_format == 'html' -%}
            {{- "\n</personality>" -}}
        {%- endif -%}

        {%- if is_vocal -%}
            {%- if prompt_format == 'markdown' -%}
                {{- "\n\n## Response Style" -}}
            {%- elif prompt_format == 'html' -%}
                {{- "\n\n<style>" -}}
            {%- endif -%}

            {{- "\n\nYou are currently in a voice call. Respond in a way that is suitable for your response to be smooth and pleasant to listen to. Prioritize short and direct responses.\n" -}}
            {{- "**ATTENTION**: user messages may contain transcription/spelling errors. Try to interpret when possible, otherwise ask to repeat." -}}

            {%- if prompt_format == 'html' -%}
                {{- "\n</style>" -}}
            {%- endif -%}
        {%- endif -%}

        {%- if python_tools or allow_code_execution -%}
            {%- if prompt_format == 'markdown' -%}
                {{- "\n\n## IPython Environment" -}}
            {%- elif prompt_format == 'html' -%}
                {{- "\n\n<ipython>" -}}
            {%- endif -%}

            {{- "\n\nYou are a Python programming expert, and you have access to a code interpreter. You can therefore write code to help you answer more complex questions, or those requiring mathematical operations.\n" -}}
            {{- "Use the `print` function to display results you want to access.\n" -}}

            {%- if python_tools -%}
                {{- "You have access to these functions that allow you to perform complex operations or search for additional information:\n```python" -}}
                {% for tool in python_tools %}
                    {{- "\n" + tool.to_signature(lang = lang) + "\n" -}}
                {% endfor %}
                {{- "```\n\n" -}}
            {%- endif -%}

            {{- "To execute code, use the standard format:\n```python\n... # your python code\n```\n" -}}
            {{- "Your code will be executed in a restricted Python environment. You only have access to standard libraries (except `os`), as well as `numpy`.\n" -}}
            {{- "\nReason step by step:" -}}
            {{- "\n1) If no function is relevant, respond directly." -}}
            {{- "\n2) If you're missing information to call a function, ask the user for it." -}}
            {{- "\n3) Otherwise call the relevant function(s) in valid Python code." -}}
            {{- "\nTest your code by displaying results (with `print`), and correct yourself if the result is not as expected!" -}}

            {%- if prompt_format == 'html' -%}
                {{- "\n</ipython>" -}}
            {%- endif -%}
        {%- endif -%}

        {%- if paragraphs -%}
            {%- if prompt_format == 'markdown' -%}
                {{- "\n\n## Information\n\n" -}}
            {%- elif prompt_format == 'html' -%}
                {{- "\n<information>\n" -}}
            {%- endif -%}

            {{- "You have access to this information to help you respond. **Use it if it's relevant**!" -}}
            {%- for para in paragraphs %}
                {{- "\n\n" -}}
                {% if 'filename' in para and (loop.index == 1 or para.filename != paragraphs[loop.index - 2].filename) -%}
                    {{- "- File: " + para.filename + "\n" -}}
                    {% if 'title' in para %}
                        {{- "- Title: " + para.title + "\n" -}}
                    {%- endif -%}
                {% elif 'url' in para and (loop.index == 1 or para.url != paragraphs[loop.index - 2].url) %}
                    {{- "- URL: " + para.url + "\n" -}}
                    {% if 'title' in para %}
                        {{- "- Title: " + para.title + "\n" -}}
                    {%- endif -%}
                {%- endif -%}

                {%- if 'page' in para and (loop.index == 1 or para.page != paragraphs[loop.index - 2].page)  -%}
                    {{- "- Page #" + para.page|string + "\n" -}}
                {%- endif -%}
                {%- if 'section' in para and para.section %}
                    {{- "- Section: " + para.section[-1] + "\n" }}
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
                {{- "\n\n## Conversation History" -}}
            {%- elif prompt_format == 'html' -%}
                {{- "\n\n<conversation>" -}}
            {%- endif -%}
            {{- "\n\nYou have access to the latest messages in the conversation. Use them to contextualize your response and reasoning.\n" -}}
        {%- endif -%}
    """),
    'format'    : dedent("""
        {%- if prefix -%}
            {{- prefix -}}
        {%- endif -%}

        {%- if is_vocal -%}
            {{- "Transcription:\n" -}}
        {%- endif -%}

        {{- text -}}
    """),
    'message_format'    : dedent("""
        {%- if message.role == 'user' and (message.user or message.time) -%}
            {%- if message.user -%}
                {{- "\n- User name: " + message.user -}}
            {%- endif -%}
            {%- if message.time -%}
                {{- "\n- Date: " + timestamp_to_str(message.time) -}}
            {%- endif -%}
            {{- "\n\n" -}}
        {%- endif -%}

        {{- text -}}
    """),
    'last_message_format'   : dedent("""
        {%- if messages and messages|length > 2 and prompt_format == 'html'  -%}
            {{- "\n</conversation>\n\n" -}}
        {%- endif -%}
        
        {%- if pinned_messages -%}
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
    """)
}

prompts_expert   = {
    'personnality_prompt' : (
        "You are an expert in {domain} with the following background: {background}\n\n",
        "Answer the questions you are asked as best as possible."
    )
}

prompts_rag    = {
    'format'    : dedent("""
        {{- "\n\nAnswer ONLY based on the information provided!\n" -}}

        {% if add_source %}
            {{- "The response must explicitly mention sources, for example by starting with `as mentioned in...`, or `According to...`." -}}
        {% endif %}

        {{- "If no information is relevant, ask the user for more details,\nIf the question is unrelated to the information, do not respond.\n" -}}
    """)
}


prompts_translate    = {
    'prefix'    : dedent("""
        {%- if not target_lang -%}
            {% set target_lang = 'French' %}
        {%- endif -%}
        
        {{- "Translate this text to " + target_lang + "\n\n" -}}
    """),
    'answer_start'  : 'The translation is:\n\n',
}

prompts_reformulate  = {
    'prefix'    : 'Rewrite this text **without explaining the changes** by improving the style, grammar and fluency:\n\n'
}

prompts_describe  = {
    'prefix'    : 'Describe in general terms the content of this text:\n\n'
}

prompts_summarize  = {
    'prefix'    : 'Write a summary of this text keeping the important information:\n\n'
}

prompts_extract_entities  = {
    'format'    : prompts_default['format'] + "\n\n" + dedent("""
        {{- "Find the requested information and make a clear and concise summary in JSON format. If information is missing, put `null`.\n\n" }}

        {{- "Here is the information to extract:\n```json\n{\n" }}
        {%- for key, value in entities.items() %}
            {{- "    \"" + key + "\": # " + value + "\n" }}
        {%- endfor %}
        {{- "}\n```\n" -}}
    """),
    'answer_start'  : "Here is the requested information:\n\n```json\n",
    'stop_words'    : '```'
}