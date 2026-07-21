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
import inspect
import logging

from loggers import timer
from utils import time_to_string

logger = logging.getLogger(__name__)

# /!\ These sentinel values are part of the public streaming contract : they are forwarded to
# `stream_callback` / `request_manager`, and **external consumers** (e.g. `bots.features`)
# redefine them locally. Do NOT change them without adapting the consumers first.
END_OF_STREAM   = None
END_OF_REQUEST  = inspect._empty

class RequestController:
    """
        Per-request lifecycle controller : abort / finalization / `request_manager` interaction,
        and rollback of the `Conversation` state when the request is aborted.

        One instance **must** be created per `infer` call (its state is request-specific).

        The `request_manager` protocol (all methods optional unless stated) :
            - `build()` : initializes the manager (called once, at controller creation)
            - `__call__(item, request_id = None) -> bool` : called at each streaming step
              (see `GenerationStream`) ; returning `False` aborts the request
            - `is_aborted(request_id) -> bool`  : polled between the pipeline steps
            - `wait_finalize(request_id) -> bool`   : **required** when `wait_finalization = True` ;
              waits until the request is finalized, returning `False` aborts it
            - `finalize(request_id)`    : called at most once, when the request completed
            - `pop(request_id)` : called when the request is aborted
    """
    def __init__(self,
                 conversation,
                 *,

                 callback   = None,
                 request_id = None,
                 request_manager    = None,
                 wait_finalization  = False
                ):
        if wait_finalization and not hasattr(request_manager, 'wait_finalize'):
            raise ValueError('`wait_finalization` requires a valid `request_manager`')

        if callback is not None and not callable(callback):
            assert hasattr(callback, 'put'), '`callback` should be callable or have a `put` method'
            callback = callback.put

        self.callback   = callback
        self.request_id = request_id
        self.request_manager    = request_manager
        self.wait_finalization  = wait_finalization

        self.conversation   = conversation
        self._initial_state = conversation.get_state() if conversation is not None else None

        if hasattr(self.request_manager, 'build'):
            self.request_manager.build()

        self._aborted   = False
        self._results   = []
        self._active_stream = None

    # Whether the generation should run in streaming mode (i.e., someone consumes the steps)
    streaming   = property(lambda self: self.request_manager is not None or self.callback is not None)

    def __len__(self):
        return len(self._results)

    def attach(self, stream, /):
        """ Registers the stream-like object to abort alongside the request """
        self._active_stream = stream
        if self._aborted and not stream.is_aborted():
            stream.abort()

    def append(self, result, /):
        """ Stores an intermediate generation result (multi-turn generation, e.g. tool calls) """
        self._results.append(result)

    def results(self):
        return self._results.copy()

    def abort(self):
        if self._aborted: return

        if self.conversation is not None:
            self.conversation.set_state(self._initial_state)

        if self.request_id is not None:
            logger.info('Request {} is aborted !'.format(self.request_id))

        self._aborted = True
        # only stream-like objects are abortable (non-streaming runtime outputs are plain data)
        if self._active_stream is not None and hasattr(self._active_stream, 'abort'):
            if not self._active_stream.is_aborted():
                self._active_stream.abort()

        if self.request_manager is not None and self.request_id is not None:
            self.request_manager.pop(self.request_id)

    def is_aborted(self):
        if self._aborted:
            return True
        elif hasattr(self.request_manager, 'is_aborted') and self.request_manager.is_aborted(self.request_id):
            self.abort()
            return True
        else:
            return False

    def update(self, item, /):
        """ Forwards a streaming step to the `request_manager` / `callback`. Returns `False` if the request was aborted """
        if self.request_manager is not None:
            if self.request_manager(item, request_id = self.request_id) is False:
                self.abort()
                return False

        if self.callback is not None:
            self.callback(item)

        return True

    def finalize(self):
        """ Finalizes the request (at most once). Returns `False` if it was aborted meanwhile """
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

class GenerationStream:
    """
        Relays a runtime generation output to a `RequestController`.

        `consume(output)` uniformizes the two runtime output flavors :
            - plain outputs (keras / non-streaming TRT-LLM) are returned as-is
            - stream-like outputs (TRT-LLM `InferenceStream`) are iterated : each step is
              decoded (`stream_text`) and forwarded to the controller (`request_manager` /
              `callback`), and the final **cumulated** tokens are returned

        Note : TRT-LLM streaming steps hold the *cumulated* `output_ids`, so `stream_text = True`
        forwards the **full decoded text so far** at each step (not the delta).
    """
    def __init__(self, controller, *, tokenizer = None, stream_text = False):
        if stream_text and tokenizer is None:
            raise ValueError('`tokenizer` is required when `stream_text = True`')

        self.controller = controller
        self.tokenizer  = tokenizer
        self.stream_text    = stream_text

        if stream_text:
            self._decode    = lambda out: tokenizer.decode(out)[0][0]
        else:
            self._decode    = lambda out: out

    def consume(self, output, /):
        if not hasattr(output, 'abort'):    # plain (non-streaming) output
            return output

        self.controller.attach(output)
        self._relay(output)
        return output.tokens

    @timer
    def _relay(self, stream):
        controller = self.controller
        if controller.is_aborted():
            if not stream.is_aborted(): stream.abort()
            return

        t0 = time.time()
        last_tokens = None
        for i, tokens in enumerate(stream):
            if controller.is_aborted():
                if not stream.is_aborted(): stream.abort()
                break

            last_tokens = tokens

            if not controller.update(self._decode(tokens)):
                if logger.isEnabledFor(logging.INFO):
                    logger.info('[LLM] Inference interrupted after {}'.format(
                        time_to_string(time.time() - t0)
                    ))
                return

            if i == 0 and logger.isEnabledFor(logging.INFO):
                logger.info('[LLM] Time-to-first token : {}'.format(
                    time_to_string(time.time() - t0)
                ))

        if last_tokens is not None and logger.isEnabledFor(logging.INFO):
            t1 = time.time()
            n  = sum(sum(len(beam) for beam in beams) for beams in last_tokens)
            logger.info('[LLM] {} tokens generated in {} ({:.3f} tokens/sec)'.format(
                n, time_to_string(t1 - t0), n / (t1 - t0)
            ))

        # sent even after an abort : consumers (e.g. `Queue`-based callbacks) rely on the
        # sentinel to unblock
        controller.update(END_OF_STREAM)
