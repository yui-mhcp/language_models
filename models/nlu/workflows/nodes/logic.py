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

import queue

from .node import NodeManager
from .iterator import ParallelIteratorNode

class AnyNode(ParallelIteratorNode):
    def run(self, context):
        iterable = self.iterable(context)
        
        if self.is_stopped() or len(iterable) == 0:
            return False
        elif len(iterable) == 1:
            context[self.item_key] = iterable[0]
            return bool(self.body(context))
        
        if not self.body.built: self.body.build()
        
        def should_stop():
            return state['valid'] or self.is_stopped()
        
        def run_and_add_to_buffer(context, /, *, index, ** kwargs):
            output = None
            try:
                output = self.body(context, ** kwargs)
            finally:
                buffer.put((index, context, output))
            return context, output
        
        state   = {'valid' : False}
        buffer  = queue.Queue()
        outputs = [
            NodeManager.run_async(
                run_and_add_to_buffer,
                {** context, self.item_key : item},
                _stopper = should_stop,
                index = i
            )
            for i, item in enumerate(iterable)
        ]
        
        results = [None] * len(outputs)
        for _ in range(len(outputs)):
            idx, ctx, out = buffer.get()
            results[idx] = out
            if out:
                state['valid'] = True
                return True
        
        return False
        
