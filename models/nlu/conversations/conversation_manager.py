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

import os
import time
import logging
import collections

from loggers import Timer, timer
from utils import load_data, dump_data, dump_json
from .message import Message
from .conversation import Conversation
from .conv_item_selectors import select_items

logger = logging.getLogger(__name__)

_inf = float('inf')
MIN_LENGTH_THRESHOLD = 10

_max_conv_in_memory = 10
_in_memory_conv_id  = '__in_memory__'
_default_selectors  = collections.OrderedDict([
    ('instructions',    {'name' : 'all'}),
    ('messages',        {'name' : 'last_messages', 'max_messages' : 5}),
    ('paragraphs',      {'name' : 'first', 'max_items' : 5})
])

class ConversationManager:
    def __init__(self):
        self._convs  = collections.OrderedDict()
        self._logged = {}   # (directory, conv_id) -> #messages already flushed to the append-log

    def get_conv(self, *, conv_id = None, directory = None, messages = None, ** kwargs):
        if conv_id is None:
            conv_id = _in_memory_conv_id
            if messages is not None: self._convs.pop((directory, conv_id), None)
        elif directory and (directory, conv_id) not in self._convs:
            loaded = load_conv(directory, conv_id)
            if loaded is not None:
                self._convs[(directory, conv_id)], self._logged[(directory, conv_id)] = loaded

        key  = (directory, conv_id)
        conv = self._convs.get(key, None)
        if conv is None:
            kwargs = {k : v for k, v in kwargs.items() if k in Conversation.__dataclass_fields__}
            kwargs['id'] = conv_id
            conv   = Conversation(** kwargs)

            if messages:
                for msg in messages: conv.add_message(msg)

        self._convs[key] = conv
        self._convs.move_to_end(key)
        self._evict_old_convs()
        return conv

    def _evict_old_convs(self):
        """
            Bounded LRU eviction of in-memory conversations.
            Only persisted (directory-backed) conversations are evicted — they are flushed to
            disk on each turn, so dropping them is lossless. The volatile in-memory session
            (`_in_memory_conv_id`) is never evicted (it is not persisted anywhere).
        """
        for key in list(self._convs.keys()):
            if len(self._convs) <= _max_conv_in_memory:
                break
            if key[1] == _in_memory_conv_id:
                continue
            self._convs.pop(key, None)
            self._logged.pop(key, None)

    @staticmethod
    @timer
    def get_context(conv,
                    query,
                    *,
                    
                    selectors   = None,
                    max_length  = None,
                    
                    documents   = None,
                    attachments = None,
                    use_specified_documents = False,
                    
                    ** kwargs
                   ):
        if selectors is None: selectors = _default_selectors.copy()
        
        for k in list(kwargs.keys()):
            if k.endswith('_selector'):
                selectors[k[:-9]] = kwargs.pop(k)
        
        if attachments and 'Attachments' not in query:
            if isinstance(attachments, str): attachments = [attachments]
            attachments = [os.path.basename(f) for f in attachments]
            query = 'Attachments:{}\n\n{}'.format(
                ' ' + attachments[0] if len(attachments) == 1 else ''.join(
                    '\n- ' + f for f in attachments
                ), query
            )
        
        
        context, remaining_length = {}, max_length
        for key, selector in selectors.items():
            if not selector:
                continue
            elif remaining_length is not None and remaining_length <= MIN_LENGTH_THRESHOLD:
                logger.warning('The remaining token budget is too small ({}), skipping {}'.format(
                    remaining_length, key
                ))
                continue
            
            max_budget = remaining_length
            
            config = kwargs.copy()
            conv_attr = key
            if isinstance(selector, dict):
                selector = selector.copy()
                if 'max_length' in selector:
                    max_budget = selector.pop('max_length')
                    if isinstance(max_budget, float): max_budget = int(max_budget * max_length)
                    max_budget = min(max_budget, remaining_length)
                
                if 'conv_attr' in selector:
                    conv_attr = selector.pop('conv_attr')
                
                if 'items' in selector:
                    config['items'] = selector.pop('items')
            
            if 'items' in config:
                pass
            elif conv_attr == 'paragraphs':
                if not use_specified_documents: documents = conv.documents
                config.update({'items' : documents, 'documents' : documents})
            elif conv_attr:
                config['items'] = getattr(conv, conv_attr).copy()
            else:
                config['items'] = selector.pop('items')

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Using selector `{}` for key `{}` (max tokens : {}, # items : {})'.format(
                    selector, key, max_budget, len(config['items']) if config['items'] else 0
                ))
            
            with Timer('{}_selector'.format(key)):
                content, length = select_items(
                    query, conv = conv, selector = selector, max_length = max_budget, ** config
                )
                if key == 'messages':
                    content = filter_multimodal_content(content, ** kwargs)
                
                context[key] = content
                if remaining_length: remaining_length -= length

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('{} {} selected'.format(len(content), key))
        
        return context
    
    def save(self, conv, directory):
        """
            Persist `conv` : append its new messages to the durable append-log, then refresh
            the (message-less) snapshot. Only the newly-added messages are written — the
            message stream is never fully rewritten (unlike the previous whole-JSON dump).
        """
        key = (directory, conv.id)
        self._logged[key] = save_conv(conv, directory, start = self._logged.get(key, 0))
        return self._logged[key]

    def close_segment(self, conv, directory, ** info):
        """
            Append a `segment_closed` marker to the durable log (segmentation seam, Yui §4.3).
            The message stream is untouched ; a worker tailing the log uses the marker as a
            batch boundary for extraction / consolidation.
        """
        append_segment_marker(conv, directory, ** info)

def get_conv_file(directory, conv_id):
    return os.path.join(directory, conv_id, 'conversation.json')

def get_log_file(directory, conv_id):
    return os.path.join(directory, conv_id, 'conversation.jsonl')

def load_conv(directory, conv_id):
    """
        Rebuild a conversation from its on-disk representation :
            - `conversation.json`   : snapshot of the conversation state (id / documents /
              instructions / metadata …), **without** the message stream
            - `conversation.jsonl`  : append-only log, authoritative for the message stream
              (and segment markers)

        Returns `(conv, n_logged)` where `n_logged` is the number of messages already durably
        in the log, or `None` if nothing is persisted for this conversation.

        Backward compatible with the legacy format (messages stored inline in the snapshot) :
        such messages are migrated to the log on the next `save_conv`. If a migration was
        interrupted, the longer of the two message lists wins, so no message is ever lost.
    """
    snap_file, log_file = get_conv_file(directory, conv_id), get_log_file(directory, conv_id)
    has_snap, has_log   = os.path.exists(snap_file), os.path.exists(log_file)
    if not has_snap and not has_log:
        return None

    conv = Conversation.load(snap_file) if has_snap else Conversation(id = conv_id)

    if not has_log:
        # legacy (or brand-new) snapshot : its inline messages are not in the log yet
        return conv, 0

    log_messages = [
        Message.from_dict(event['data'])
        for event in load_data(log_file) if event.get('type') == 'message'
    ]
    if len(conv.messages) > len(log_messages):
        # interrupted migration : the snapshot still holds the (longer) inline stream
        return conv, len(log_messages)

    conv.messages = log_messages
    return conv, len(log_messages)

def save_conv(conv, directory, *, start = 0):
    """
        Append the messages `conv.messages[start:]` to the durable log, then refresh the
        message-less snapshot (atomically). Returns the new number of logged messages.
    """
    if not directory or conv.id == _in_memory_conv_id:
        return start

    new_messages = conv.messages[start:]
    if new_messages:
        dump_data(get_log_file(directory, conv.id), [
            {'type' : 'message', 'data' : message.to_json()} for message in new_messages
        ], mode = 'a')

    _atomic_dump_json(get_conv_file(directory, conv.id), conv.get_snapshot())
    return len(conv.messages)

def append_segment_marker(conv, directory, ** info):
    """ Append a `segment_closed` event to the durable log (see `ConversationManager.close_segment`). """
    if not directory or conv.id == _in_memory_conv_id:
        return
    dump_data(get_log_file(directory, conv.id), [{
        'type' : 'segment_closed',
        'data' : {'time' : time.time(), 'n_messages' : len(conv.messages), ** info}
    }], mode = 'a')

def _atomic_dump_json(filename, data):
    """ Crash-safe JSON write : dump to a temporary file then atomically replace the target. """
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    tmp = filename + '.tmp'
    dump_json(tmp, data, indent = 2)
    os.replace(tmp, filename)

def filter_multimodal_content(_messages,
                              /,
                              *,
                              
                              max_image = None,
                              max_audio = None,
                              max_video = None,
                              max_screenshot    = None,
                              
                              keep_last_n_image = None,
                              keep_last_n_audio = None,
                              keep_last_n_video = None,
                              keep_last_n_screenshot    = 1,
                              ** _
                             ):
    filters = {
        'max_audio' : max_audio or _inf,
        'max_image' : max_image or _inf,
        'max_video' : max_video or _inf,
        'max_screenshot'    : max_screenshot or _inf,
        
        'keep_last_n_audio' : keep_last_n_audio or _inf,
        'keep_last_n_image' : keep_last_n_image or _inf,
        'keep_last_n_video' : keep_last_n_video or _inf,
        'keep_last_n_screenshot'    : keep_last_n_screenshot or _inf
    }
    if all(v == _inf for v in filters.values()): return _messages
    
    count = {'audio' : 0, 'image' : 0, 'video' : 0, 'screenshot' : 0}
    
    n = 0
    items = _messages.copy()
    for i in reversed(range(len(items))):
        item = items[i]
        if item['role'] == 'user': n += 1
        if isinstance(item['content'], str):
            continue
        
        _rm_index = []
        for j, c in enumerate(item['content']):
            if c.get('type', 'text') == 'text':
                continue
            elif n >= filters['keep_last_n_{}'.format(c['type'])]:
                _rm_index.append(j)
            elif count[c['type']] >= filters['max_{}'.format(c['type'])]:
                _rm_index.append(j)
            elif c.get('screenshot', False) and (
                n >= filters['keep_last_n_screenshot'] or count['screenshot'] >= filters['max_screenshot']
            ):
                _rm_index.append(j)
            else:
                count[c['type']] += 1
                if c.get('screenshot', False): count['screenshot'] += 1

        if _rm_index:
            item = item.copy() if isinstance(item, dict) else item.to_json()
            item['content'] = item['content'].copy()
            for idx in reversed(_rm_index): item['content'].pop(idx)
            items[i] = item
    
    return items