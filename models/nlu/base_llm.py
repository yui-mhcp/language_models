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
import logging
import warnings

from functools import partial

from loggers import timer
from utils import FunctionCallback, QueueCallback, is_dataframe, pad_batch

from ..core import BaseModel, TextModelMixin
from .streaming import RequestController, GenerationStream
from .prompts import PromptFormatter, add_template_wrapper
from .conversations import ConversationManager

logger = logging.getLogger(__name__)

class BaseLLM(TextModelMixin, BaseModel):
    """
        Base class for (possibly multimodal) text-generation LLMs.

        A prediction goes through the following steps (see `infer`) :
            1) Session resolution   : `Conversation` retrieval + saving directories
            2) Prompt preparation   : query formatting (`PromptFormatter`), context selection
               (`ConversationManager.get_context`), chat-template rendering, tokenization
               and, for multimodal models, features preparation (`prepare_multimodal_data`)
            3) Generation   : `compiled_infer` on the runtime ; when streaming, the steps are
               relayed to `stream_callback` / `request_manager` (see `streaming.py`)
            4) Finalization : conversation update / persistence, inference callbacks

        The main runtime is **TensorRT-LLM** (streaming, guided decoding, stop conditions) ;
        the keras runtime supports basic generation only (see the `Runtime` capabilities).

        Note : the tool-calling / code-execution loop was quarantined in `_tool_calling.py`
        (broken, pending repair) ; `infer` raises when `tools` / `allow_code_execution` is set.
    """
    _directories    = {
        ** BaseModel._directories,
        'conv_dir'      : '{root}/{self.name}/conversations'
    }

    _default_pretrained_model   = None

    _default_loss   = 'TextLoss'
    _default_metrics    = 'TextAccuracy'

    input_signature     = TextModelMixin.text_signature
    output_signature    = TextModelMixin.text_signature
    decode_output   = TextModelMixin.decode_text

    def __init__(self,
                 lang,
                 *,

                 input_format   = None,
                 max_input_length   = None,
                 kv_cache_enable_block_reuse    = True,

                 ** kwargs
                ):
        pretrained = kwargs.pop('pretrained', self._default_pretrained_model)
        if 'tokenizer' not in kwargs and pretrained:
            kwargs['tokenizer'] = pretrained

        self._init_text(lang = lang, ** kwargs)

        self.input_format   = input_format
        self._max_input_length  = max_input_length

        if isinstance(pretrained, str): kwargs.setdefault('pretrained_name', pretrained)
        super().__init__(
            pretrained = pretrained, kv_cache_enable_block_reuse = kv_cache_enable_block_reuse,
            ** kwargs
        )

        self.kv_cache_enable_block_reuse    = kv_cache_enable_block_reuse

        if hasattr(self.model, 'set_tokens'): self.model.set_tokens(** self.model_tokens)

        self.conv_manager   = ConversationManager()
        self.prompt_formatter   = PromptFormatter(
            self.tokenizer,
            audio_token = getattr(self, 'audio_token', None),
            image_token = getattr(self, 'image_token', None),
            video_token = getattr(self, 'video_token', None)
        )

    def build(self, *, model = None, pretrained = None, ** kwargs):
        if model is not None: return super().build(model = model)

        if pretrained is not None:
            from architectures.transformers import get_pretrained_transformer

            super().build(
                model = get_pretrained_transformer(pretrained, ** kwargs)
            )
        else:
            super().build(** kwargs)

    @property
    def max_input_length(self):
        return self._max_input_length or getattr(self.model, 'max_input_length', None)

    @property
    def max_output_length(self):
        return getattr(self.model, 'max_output_length', None)

    @property
    def supports_mm_features_caching(self):
        return hasattr(self, 'cache_multimodal_features')

    @property
    def default_metrics_config(self):
        return {
            'pad_value' : self.blank_token_idx,
            'eos_value' : self.eos_token_idx,
            'decode_fn' : partial(self.decode_text, remove_tokens = True)
        }

    def __str__(self):
        des = super().__str__()
        des += self._str_text()
        if self.input_format:
            des += "- Input format : {}\n".format(self.input_format)

        des += "- Max input length : {}\n".format(self.max_input_length)

        return des

    def prepare_input(self, data = None, ** kwargs):
        if is_dataframe(data): data = data.to_dict('records')
        if isinstance(data, list):
            return [self.prepare_input(d, ** kwargs) for d in data]

        input_format = kwargs.pop('format', self.input_format)

        if data is None:            data = kwargs
        elif isinstance(data, str): data = {** kwargs, 'text' : data}
        elif kwargs:    data = {** kwargs, ** data}

        return self.prepare_text(data, format = input_format, ** kwargs)

    @timer
    @add_template_wrapper('default')
    def infer(self,
              text  = None,
              *,

              documents = None,
              max_input_length  = None,

              tools = None,
              allow_code_execution  = False,

              stop_words  = None,
              max_new_tokens  = 2048,
              possible_answers  = None,

              add_answer_start    = True,

              request_id    = None,
              stream_text   = True,
              request_manager   = None,
              stream_callback   = None,
              wait_finalization = False,

              save  = None,
              conv  = None,
              conv_id   = None,
              directory = None,

              callbacks = None,
              predicted = None,

              ** kwargs
             ):
        """
            Generates the LLM answer for the given `text` query.

            Arguments :
                - text  : the input query. If `None`, only pre-computes the multimodal
                  features (see the note below)

                - documents : documents added to the conversation (RAG)
                - max_input_length  : maximum number of input tokens. A negative value is a
                  margin subtracted from the model's own limit

                - stop_words    : word(s) that stop the generation once generated
                - max_new_tokens    : maximum number of tokens to generate
                - possible_answers  : restricts the generation to the given answers
                  (guided decoding)

                - add_answer_start  : whether to prepend `answer_start` (the forced start of
                  generation) to the output text

                - stream_text   : whether to forward decoded text (the **full** text so far at
                  each step) or raw tokens to `stream_callback` / `request_manager`
                - request_id / request_manager  : per-request control (abort / finalization) ;
                  see the `RequestController` docstring for the manager protocol
                - stream_callback   : `callable` (or object with a `put` method) called at each
                  generation step
                - wait_finalization : whether `finalize` waits for the `request_manager`

                - save / conv / conv_id / directory : conversation persistence :
                    {directory}/{conv.id}/
                        multimodal/         # only for multimodal models
                            features/       # cached multimodal features
                                {id}.npy
                                map.json    # maps the data path to its feature file
                        conversation.jsonl  # append-only message/segment log (durable, worker-tailable)
                        conversation.json   # snapshot of the conversation state (minus the message stream)

                - callbacks / predicted : see `BaseModel.infer`

                - kwargs    : forwarded to the prompt preparation and the runtime
            Return :
                - output    : `dict` with the `predicted` text + general information

            Note : `text = None` pre-computes and caches the multimodal features (`kwargs`
            holds the data), returning a `bool` (supported or not). This side-channel is part
            of the external API and will be replaced by a dedicated method later on.
        """
        if tools or allow_code_execution:
            raise NotImplementedError(
                'The tool-calling / code-execution support is currently broken and quarantined '
                'in `models/nlu/_tool_calling.py` ; it is not available through `infer`'
            )

        ##############################
        #     Session resolution     #
        ##############################

        if save is None: save = conv_id is not None or conv is not None

        if save:
            if conv_id is None:     conv_id = 'cli'
            if directory is None:   directory = self.conv_dir

        if conv is None:
            conv = self.conv_manager.get_conv(
                conv_id = conv_id, directory = directory, ** kwargs
            )

        conv_dir, mm_conv_dir = None, None
        if save:
            conv_dir    = os.path.join(directory, conv.id)
            mm_conv_dir = os.path.join(conv_dir, 'multimodal')
            os.makedirs(conv_dir, exist_ok = True)

        if text is None:
            if not self.supports_mm_features_caching:
                return False

            self.cache_multimodal_features(directory = mm_conv_dir, ** kwargs)
            return True

        max_input_length = self._resolve_max_input_length(max_input_length)

        controller  = RequestController(
            conv,
            callback    = stream_callback,
            request_id  = request_id,
            request_manager = request_manager,
            wait_finalization   = wait_finalization
        )
        # runtime-specific generation kwargs, kept out of `kwargs` (which flows to the prompt
        # preparation and the output `dict`)
        runtime_kwargs  = self._get_generation_config(
            possible_answers = possible_answers, stop_words = stop_words
        )

        ##############################
        #     Prompt preparation     #
        ##############################

        query = self.prompt_formatter.prepare_query(text, ** kwargs)

        if documents:
            kwargs['attachments'] = conv.add_document(documents)

        context = self.conv_manager.get_context(
            conv,
            query,
            tokenizer   = self.tokenizer,
            directory   = conv_dir,
            documents   = documents,
            max_length  = max_input_length,
            ** kwargs
        )

        context.setdefault('messages', []).append(
            conv.add_message(text = query, role = 'user', ** kwargs)
        )

        prompt, multimodal_data = self.prompt_formatter.get_prompt(
            allow_code_execution = allow_code_execution, ** {** kwargs, ** context}
        )
        tokens = self.tokenizer.encode(prompt, add_eos = False, return_type = 'np')

        if controller.is_aborted():
            return {}

        if multimodal_data:
            if not hasattr(self, 'prepare_multimodal_data'):
                raise NotImplementedError('{} is not multimodal'.format(self.name))

            tokens, multimodal_inputs = self.prepare_multimodal_data(
                tokens, directory = mm_conv_dir, ** multimodal_data
            )
            runtime_kwargs.update(multimodal_inputs)

        if controller.is_aborted():
            return {}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Input prompt :\n{}'.format(prompt))

        ####################
        #     Inference    #
        ####################

        out = self.compiled_infer(
            tokens[None],
            tokenizer   = self.tokenizer,
            streaming   = controller.streaming,
            max_new_tokens  = max_new_tokens,
            ** {** kwargs, ** runtime_kwargs}
        )
        out = GenerationStream(
            controller, tokenizer = self.tokenizer, stream_text = stream_text
        ).consume(out)

        if controller.is_aborted():
            return {}

        pred = self.decode_output(out)[0]
        if isinstance(pred, list):
            if len(pred) > 1:
                warnings.warn('Multiple outputs have been generated, which is not supported yet. Only the 1st one will be returned')
            pred = pred[0]

        if add_answer_start and kwargs.get('answer_start', None):
            pred = kwargs['answer_start'] + pred

        ####################
        #   Finalization   #
        ####################

        conv.add_message(text = pred, role = 'assistant', prompt = prompt)

        result = {
            'predicted' : pred,

            'conv'  : conv,
            'query' : query,
            'format'    : kwargs.get('format', None),
            'prompt'    : prompt,
            'input_tokens'  : tokens,
            ** kwargs
        }

        if not controller.finalize():
            return result
        elif callbacks:
            # `save = False` : persistence is handled by `conv_manager` below, not by
            # file-saving callbacks (no `JSONSaver` / `predicted` mapping for LLMs)
            self._finalize_predictions(
                result, callbacks = callbacks, predicted = predicted, save = False
            )

        if save:
            self.conv_manager.save(conv, directory = directory)

        return result

    def _resolve_max_input_length(self, max_input_length):
        """ Resolves the effective input token budget (negative value = margin below the model's limit) """
        if max_input_length is None:
            max_input_length = float('inf')
        if self.max_input_length:
            if max_input_length > 0:
                max_input_length = min(max_input_length, self.max_input_length)
            else:
                max_input_length = self.max_input_length + max_input_length
        return max_input_length

    def _get_generation_config(self, *, possible_answers = None, stop_words = None):
        """
            Builds the runtime-specific generation kwargs (guided decoding / stop condition),
            validating the runtime capabilities (see `Runtime`)
        """
        config = {}
        if possible_answers:
            if not self.model.supports_guided_decoding:
                raise NotImplementedError(
                    '`possible_answers` requires a runtime supporting guided decoding '
                    '(e.g. `TensorRT-LLM`), got `{}`'.format(self.runtime)
                )

            config['allowed_tokens'] = pad_batch(
                self.encode_text(
                    possible_answers, add_sos_and_eos = False, return_type = 'np'
                ),
                pad_value = self.blank_token_idx,
                dtype = 'int32'
            )

        if stop_words:
            if not self.model.supports_stop_condition:
                raise NotImplementedError(
                    '`stop_words` requires a runtime supporting stop conditions '
                    '(e.g. `TensorRT-LLM`), got `{}`'.format(self.runtime)
                )

            if isinstance(stop_words, str): stop_words = [stop_words]
            config['stop_condition'] = lambda text: any(w in text for w in stop_words)

        return config

    def get_inference_callbacks(self, *, post_processing = None, ** kwargs):
        predicted, callbacks = {}, []

        if post_processing is not None:
            if not isinstance(post_processing, list): post_processing = [post_processing]
            for fn in post_processing:
                if callable(fn):
                    callbacks.append(FunctionCallback(fn))
                elif hasattr(fn, 'put'):
                    callbacks.append(QueueCallback(fn))

        return predicted, callbacks

    def filter_prediction_output(self, output):
        return {k : v for k, v in output.items() if k not in ('conv', 'prompt', 'input_tokens')}

    def get_config(self):
        return {
            ** super().get_config(),
            ** self.get_config_text(),
            'input_format'  : self.input_format,
            'max_input_length'  : self._max_input_length
        }

    answer  = add_template_wrapper('answer', fn = infer)

    ask_expert  = add_template_wrapper('expert',      fn = answer)
    translate   = add_template_wrapper('translate',   fn = answer)
    reformulate = add_template_wrapper('reformulate', fn = answer)
    describe    = add_template_wrapper('describe',    fn = answer)
    summarize   = add_template_wrapper('summarize',   fn = answer)
    extract_entities    = add_template_wrapper('extract_entities', fn = answer)
