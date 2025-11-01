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

def select_last_messages(query,
                         items,
                         *,
                         
                         tokenizer,
                         
                         max_length = None,
                         min_messages   = 1,
                         max_messages   = 5,
                         
                         ** _
                        ):
    """
        Return a `list` of `Message` from the given `conv`

        Arguments :
            - query : the user query (not used)
            - items : the `list` of `Message` from which to select

            - tokenizer : the model tokenizer (used to compute messages' length in tokens)

            - max_length    : maximum cumulated length (in tokens)
            - min_messages  : minimum number of user messages
            - max_messages  : maximum number of user messages

        Note :
        If `min_messages < 1`, it is possible that no user message is selected
        On the other hand, if `min_messages >= 1`, the cumulated length may exceeds `max_length`.
        In the second case, a warning is displayed.
    """
    if not max_length: max_length = float('inf')

    messages, n, total_length = [], 0, 0
    for message in reversed(items):
        if message['content_type'] != 'text' or message['role'] == 'system':
            continue
        
        if 'length' not in message or not message['length']:
            message['length'] = len(tokenizer.tokenize(message['content']))

        if messages and n >= min_messages and total_length + message['length'] > max_length:
            break

        total_length += message['length']
        messages.append(message)

        if message['role'] == 'user':
            n += 1
            if n == max_messages:
                break

    return messages[::-1]

