# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
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

from tqdm import tqdm as tqdm_progress_bar

from loggers import timer
from utils.search.vectors import build_vectors_db
from utils import is_dataframe, load_json, dump_json, create_stream, pad_batch
from utils.keras_utils import ops
from utils.text import Conversation, parse_document
from .prompts import add_prompt_wrapper
from .base_language_model import BaseLanguageModel

logger = logging.getLogger(__name__)

class TextGenerator(BaseLanguageModel):
    _directories    = {
        ** BaseLanguageModel._directories,
        'vectors_dir'   : '{root}/{self.name}/vectors',
        'conv_dir'      : '{root}/{self.name}/conversations'
    }

    _default_pretrained_model   = None
    
    _default_loss   = 'TextLoss'
    _default_metrics    = 'TextAccuracy'
    
    output_signature    = BaseLanguageModel.text_signature
    
    def __init__(self, * args, output_format = None, max_output_length = None, ** kwargs):
        self.output_format  = output_format
        self._max_output_length = max_output_length

        super().__init__(* args, ** kwargs)
        
        if not self.output_format:
            self.output_format = self.input_format
        
        self._databases = {}
        self._conversations = {}
    
    @property
    def max_output_length(self):
        if self._max_output_length: return self._max_output_length
        elif hasattr(self.model, 'max_output_length'):
            return self.model.max_output_length
        return None
    
    @property
    def input_signature(self):
        return (self.text_signature, self.text_signature)

    @property
    def training_hparams(self):
        return super().training_hparams(
            max_output_length   = None,
            teacher_forcing_eval    = True,
            eval_infer_config   = {},
            show_input  = None
        )

    def infer(self, * args, ** kwargs):
        if self.max_output_length:
            kwargs.setdefault('max_length', self.max_output_length)
        
        return super().infer(* args, ** kwargs)
    
    def prepare_data(self, data):
        inputs, outputs = super().encode_data(data)

        if self.is_encoder_decoder:
            if not isinstance(outputs, tuple):
                inputs, outputs = (inputs, outputs[:-1]), outputs[1:]
            else:
                inputs, outputs = (inputs, outputs[0][:-1]), (outputs[0][1:], ) + outputs[1:]
        elif isinstance(outputs, tuple):
            if self.sos_token: outputs = (outputs[0][1:], ) + outputs[1:]
            if self.eos_token: inputs  = inputs[:-1]
            inputs = ops.concat([inputs, outputs[0]], axis = -1)
        else:
            if self.sos_token: outputs = outputs[1:]
            if self.eos_token: inputs  = inputs[:-1]
            inputs = ops.concat([inputs, outputs], axis = -1)

        return inputs, outputs
    
    def filter_output(self, output):
        if isinstance(outputs, tuple): output = output[0]
        return ops.logical_and(
            ops.all(ops.shape(output) > 0), 
            ops.shape(output)[-1] <= self.max_output_length
        )

    def get_conversation(self, id = None, directory = None):
        if not id: id = 'default'
        
        if id not in self._conversations:
            if not directory: directory = self.conv_dir
            os.makedirs(directory, exist_ok = True)
            
            if id + '.json' not in os.listdir(directory):
                conv = Conversation(id = id)
            else:
                conv = Conversation.load(os.path.join(directory, id + '.json'))
            self._conversations[id] = conv
        
        return self._conversations[id]
        
    def get_vectors_database_file(self, *, name = None, filename = None, directory = None, ** _):
        if name and os.path.isfile(name):
            filename = name
        else:
            if not directory:   directory = self.vectors_dir
            if not filename:    filename = '{}.h5'.format(name or 'database')

            filename = os.path.join(directory, filename)
        return filename, os.path.basename(filename).split('.')[0]
    
    def add_vectors_database(self, vectors, ** kwargs):
        file, name = self.get_vectors_database_file(** kwargs)
        self._databases[name] = vectors
    
    def get_vectors_database(self, ** kwargs):
        file, name = self.get_vectors_database_file(** kwargs)

        if self._databases.get(name, None) is None:
            self._databases[name] = build_vectors_db(file)
        return self._databases[name]
    
    @timer
    def predict(self,
                texts,
                batch_size = 16,
                *,
                
                format  = None,
                
                stream_text = False,
                stream_callback = None,
                
                stop_words  = None,
                max_input_len   = None,
                max_new_tokens  = 512,
                add_answer_start    = True,
                
                save    = False,
                directory   = None,
                overwrite   = False,
                
                chat_id = None,
                chat_mode   = None,
                chat_filter = 5,
                
                method  = 'beam',
                num_beams   = 5,
                num_sentences   = 1,
                
                tqdm    = 'auto',
                
                ** kwargs
               ):
        if not max_input_len: max_input_len = self.max_output_length
        elif self.max_input_length: max_input_len = min(max_input_len, self.max_output_length)
        if chat_mode is None: chat_mode = chat_id is not None
        if is_dataframe(texts):              texts = texts.to_dict('records')
        elif isinstance(texts, (str, dict)): texts = [texts]
        if tqdm == 'auto': tqdm = tqdm_progress_bar if len(texts) > 1 else lambda x: x
        
        now = time.time()
        
        if not save and not chat_id: chat_id = '__unsave__'
        
        conv     = self.get_conversation(chat_id, directory = directory)
        messages = []
        if chat_mode:
            if not chat_filter:
                messages = conv.filter(** kwargs)
            elif isinstance(chat_filter, int):
                messages = conv.last_conv[- chat_filter * 2 :]
            elif isinstance(chat_filter, dict):
                messages = conv.filter(** {** kwargs, ** chat_filter})
            else:
                raise ValueError('Unsupported filter : {}'.format(conv_filter))
            
            if logger.isEnabledFor(logging.DEBUG) and messages:
                logger.debug('Messages : {}'.format(messages))
        
        if format:
            texts = [
                self.text_encoder.apply_format(format, text = text, role = 'user', ** kwargs)
                for text in texts
            ]

        inputs = self.get_input(texts, messages = messages, max_length = max_input_len, ** kwargs)
        input_texts = self.decode_text(inputs)
        
        if stop_words:
            if isinstance(stop_words, str):
                stop_words = [self.encode_text(
                    stop_words, add_sos = False, add_eos = False, return_type = 'list'
                )]
            elif isinstance(stop_words[0], str):
                stop_words = self.encode_text(
                    stop_words, add_sos = False, add_eos = False, return_type = 'list'
                )
            elif isinstance(stop_words[0], int):
                stop_words = [stop_words]
            
            kwargs['stop_words_list'] = [stop_words]
        
        kwargs.update({
            'max_input_len' : max_input_len,
            'max_new_tokens'    : max_new_tokens,
        })
        if stream_text: kwargs['decode_fn'] = self.decode_output
        
        results = []
        for i in tqdm(range(0, len(inputs), batch_size)):
            out = self.compiled_infer(
                inputs[i : i + batch_size], stream_callback = stream_callback, ** kwargs
            )
            out = self.decode_output(out)

            for j, (inp, pred) in enumerate(zip(texts[i * batch_size : (i + 1) * batch_size], out)):
                if isinstance(pred, list) and len(pred) == 1: pred = pred[0]
                if kwargs.get('answer_start', None) and add_answer_start:
                    pred = kwargs['answer_start'] + pred
                
                conv.append(
                    inp, role = 'user', time = now, new_conv = not messages, ** kwargs
                )
                conv.append(
                    pred, role = 'assistant', infos = {
                        ** kwargs,
                        'user'  : None,
                        'user_id'   : None,
                        'query' : inp,
                        'input' : input_texts[i + j],
                        'input_tokens'  : inputs[i + j]
                    }
                )
                results.append(conv[-1])

        if save:
            conv.save(os.path.join(directory or self.conv_dir, str(conv.id) + '.json'))

        return results
    
    @add_prompt_wrapper('answer')
    def answer(self, question, *, possible_answers = None, documents = None, ** kwargs):
        if possible_answers:
            kwargs['allowed_tokens'] = pad_batch(
                self.encode_text(
                    possible_answers, add_eos = False, add_sos = False, return_type = 'np'
                ),
                pad_value = self.blank_token_idx,
                dtype = 'int32'
            )
        
        if documents:
            if not isinstance(documents, (list, tuple)): documents = [documents]
            
            parsed = []
            for doc in documents:
                if isinstance(doc, str):
                    parsed.extend(parse_document(doc, ** kwargs))
                elif isinstance(doc, dict):
                    parsed.append(doc)
                elif isinstance(doc, list):
                    parsed.extend(doc)
                else:
                    raise ValueError('Unsupported document format : {}'.format(doc))
            documents = parsed
        
        if not question:
            assert documents, 'You must provide a question or documents'
            question = '\n\n'.join([para['text'].strip('\n') for para in documents if 'text' in para])
            documents = None
        
        return self.predict(question, documents = documents, ** kwargs)

    translate   = add_prompt_wrapper('translate', fn = answer)
    reformulate = add_prompt_wrapper('reformulate', fn = answer)
    describe    = add_prompt_wrapper('describe', fn = answer)
    summarize   = add_prompt_wrapper('summarize', fn = answer)
    extract_entities    = add_prompt_wrapper('extract_entities', fn = answer)

    @add_prompt_wrapper('rag')
    def rag(self,
            question,
            *,
            
            k   = 10,
            reverse = True,
            informations    = None,
            
            vectors = None,
            retriever   = None,
            documents   = None,
            retriever_config    = {},
            
            search_on_web   = None,
            search_config   = {},
               
            ** kwargs
           ):
        if search_on_web is None: search_on_web = documents is None and vectors is None
        
        parsed = []
        if documents:
            if not isinstance(documents, (list, tuple)): documents = [documents]
            
            for doc in documents:
                if isinstance(doc, str):
                    parsed.extend(parse_document(doc, ** kwargs))
                elif isinstance(doc, dict):
                    parsed.append(doc)
                elif isinstance(doc, list):
                    parsed.extend(doc)
                else:
                    raise ValueError('Unsupported document format : {}'.format(doc))
        
        if search_on_web:
            from utils.search.web import search
            
            for doc in search(question, ** kwargs)['results']:
                parsed.extend(doc)
        
        if retriever is not None:
            if isinstance(retriever, str):
                from models import get_pretrained
                retriever = get_pretrained(retriever)

            retriever_config = {
                ** retriever_config, 'save' : retriever_config.get('save', True)
            }
            vectors = retriever.predict(parsed, to_numpy = True, ** retriever_config)
        
        if vectors is None:
            raise RuntimeError('`vectors` is None, which is unexpected for a RAG method')
        
        if isinstance(vectors, str):
            vectors = self.get_vectors_database(name = vectors, ** kwargs)

        if not informations:                     informations = [question]
        elif not isinstance(informations, list): informations = [informations]
        
        retrieved, _texts = [], set()
        for info in informations:
            if not isinstance(info, dict): info = {'query' : info}
            retrieved_info = vectors.search(info['query'], k = k, reverse = reverse)
            
            if 'question' in info:
                retrieved_info = [{'text' : self.answer(
                    info['question'], documents = retrieved_info
                )[0]}]
            
            for ret in retrieved_info:
                if ret['text'] not in _texts:
                    retrieved.append(ret)
                    _texts.add(ret['text'])

        return self.answer(question, documents = retrieved, ** kwargs)

    def stream_fn(self,
                  text,
                  task      = 'answer',
                  chat_mode = True,
                  use_tree_reasoning = False,
                  ** kwargs
                 ):
        if not hasattr(self, task):
            raise ValueError('The task `{}` does not exist'.format(task))
        elif task != 'predict' and not hasattr(getattr(self, task), 'prompt_key'):
            raise ValueError('`{}` is not a prediction method !'.format(task))
        
        if isinstance(text, list): text = ' '.join(text)
        if use_tree_reasoning:
            raise NotImplementedError('Work in progress !')
        
        return getattr(self, task)(text, chat_mode = chat_mode, ** kwargs)[0]
    
    def stream(self, stream, ** kwargs):
        return create_stream(self.stream_fn, stream = stream, ** kwargs)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'output_format' : self.output_format,
            'max_output_length' : self.max_output_length
        })
        return config

