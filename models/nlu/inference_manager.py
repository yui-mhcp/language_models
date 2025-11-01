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

import time
import logging
import inspect

from loggers import timer
from utils import time_to_string

logger = logging.getLogger(__name__)

END_OF_STREAM   = None
END_OF_REQUEST  = inspect._empty

class InferenceManager:
    def __init__(self,
                 initial_state,
                 *,
                 
                 tokenizer  = None,
                 stream_text    = False,
                 
                 callback   = None,
                 request_id = None,
                 request_manager    = None,
                 wait_finalization  = False
                ):
        if wait_finalization and not hasattr(request_manager, 'wait_finalize'):
            raise VaueError('`wait_finalization` requires a valid `request_manager`')
        
        if stream_text and tokenizer is None:
            raise ValueError('`tokenizer` is required when `stream_text = True`')
        
        if callback is not None and not callable(callback):
            assert hasattr(callback, 'put'), '`callback` should be callable or have a `put` method'
            callback = callback.put
        
        self.tokenizer  = tokenizer
        self.stream_text    = stream_text
        
        self.callback   = callback
        self.request_id = request_id
        self.request_manager    = request_manager
        self.wait_finalization  = wait_finalization
        
        self._initial_state = initial_state
        
        if hasattr(self.request_manager, 'build'):
            self.request_manager.build()
        
        self._aborted   = False
        self._all_results   = []
        self._inference_stream  = None
        
        if self.stream_text:
            self._decode    = lambda out: tokenizer.decode(out)[0][0]
        else:
            self._decode    = lambda out: out
        self._streaming = self.request_manager is not None or self.callback is not None
        self._inference_config  = {'streaming' : self._streaming}
    
    @property
    def stream(self):
        return self._inference_stream
    
    @stream.setter
    def stream(self, value):
        self._inference_stream = value
    
    @property
    def initial_state(self):
        return self._initial_state
    
    def __len__(self):
        return len(self._all_results)
    
    def get_inference_config(self):
        return self._inference_config.copy()
    
    def set_inference_stream(self, stream, /):
        self.stream = stream
        
        if self._streaming: self.start_stream()
    
    def abort(self):
        if self._aborted: return
        
        if self.request_id is not None:
            logger.info('Request {} is aborted !'.format(self.request_id))
        
        self._aborted = True
        if self.stream is not None and not self.stream.is_aborted():
            self.stream.abort()
        
        if self.request_manager is not None and self.request_id is not None:
            self.request_manager.pop(self.request_id)
    
    def is_aborted(self):
        if self._aborted:
            return True
        elif hasattr(self.request_manager, 'is_aborted')and self.request_manager.is_aborted(self.request_id):
            self.abort()
            return True
        else:
            return False

    @timer
    def start_stream(self):
        if self.is_aborted() and not self.stream.is_aborted():
            self.stream.abort()
            return
        
        t0 = time.time()
        for i, tokens in enumerate(self.stream):
            if self.is_aborted() and not self.stream.is_aborted():
                self.stream.abort()
                break
            
            out = self._decode(tokens)
            
            if self.request_manager is not None:
                if self.request_manager(out, request_id = self.request_id) is False:
                    self.abort()
                    if logger.isEnabledFor(logging.INFO):
                        logger.info('[LLM] Inference interrupted after {}'.format(
                            time_to_string(time.time() - t0)
                        ))
                    return
                
            if self.callback is not None:
                self.callback(out)
            
            if i == 0 and logger.isEnabledFor(logging.INFO):
                t1 = time.time()
                logger.info('[LLM] Time-to-first token : {}'.format(
                    time_to_string(t1 - t0)
                ))

        if logger.isEnabledFor(logging.INFO):
            t1 = time.time()
            n  = sum(sum(len(beam) for beam in beams) for beams in tokens)
            logger.info('[LLM] {} tokens generated in {} ({:.3f} tokens/sec)'.format(
                n, time_to_string(t1 - t0), n / (t1 - t0)
            ))

        if self.request_manager is not None:
            if self.request_manager(END_OF_STREAM, request_id = self.request_id) is False:
                self.abort()
                return

        if self.callback is not None:
            self.callback(END_OF_STREAM)
    
    def append(self, result, /):
        self._all_results.append(result)
    
    def result(self):
        if hasattr(self.stream, 'result'):
            return [out.token_ids for out in self.stream.result().outputs]
        else:
            return self.stream
    
    def cumulated_results(self):
        return self._all_results.copy()
    
    def finalize(self):
        if self.callback is not None:
            self.callback(END_OF_REQUEST)
        
        if self.request_manager is not None:
            if self.wait_finalization and not self.request_manager.wait_finalize(self.request_id):
                self.abort()
                return False

            if hasattr(self.request_manager, 'finalize'):
                self.request_manager.finalize(self.request_id)
        
        if self.request_id is not None:
            logger.info('Request {} is finished !'.format(self.request_id))

        return True
