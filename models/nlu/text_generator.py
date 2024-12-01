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
import logging

from tqdm import tqdm as tqdm_progress_bar

from loggers import timer
from utils.search.vectors import build_vectors_db
from utils import is_dataframe, load_json, dump_json
from utils.keras_utils import ops
from utils.text import parse_document
from .prompts import add_prompt_wrapper
from .base_language_model import BaseLanguageModel

logger = logging.getLogger(__name__)

class TextGenerator(BaseLanguageModel):
    _default_pretrained_model   = 'facebook/bart-large'
    
    _default_loss   = 'TextLoss'
    _default_metrics    = 'TextAccuracy'
    
    output_signature    = BaseLanguageModel.text_signature
    
    def __init__(self, * args, output_format = None, max_output_length = None, ** kwargs):
        self.output_format  = output_format
        self.max_output_length = max_output_length

        super().__init__(* args, ** kwargs)
        
        if not self.output_format:
            self.output_format = self.input_format
        
        self._databases = {}
    
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

    def get_vectors_database_file(self, *, name = None, filename = None, directory = None, ** _):
        if not directory:   directory = self.pred_dir
        if not filename:    filename = '{}.h5'.format(name or 'database')
        
        filename = os.path.join(directory, filename)
        return filename, os.path.basename(filename).split('.')[0]
    
    def add_vectors_database(self, vectors, ** kwargs):
        file, name = self.get_vectors_database_file(** kwargs)
        vectors.save(file)
        self._databases[name] = vectors
    
    def get_vectors_database(self, ** kwargs):
        file, name = self.get_vectors_database_file(** kwargs)
        
        if name not in self._databases:
            self._databases[name] = build_vectors_db(file)
        return self._databases[name]
    
    @timer
    def predict(self,
                texts,
                batch_size = 16,
                *,
                
                max_input_len   = None,
                max_new_tokens  = 256,
                
                save    = False,
                directory   = None,
                overwrite   = False,
                filename    = 'map.json',
                
                method  = 'beam',
                num_beams   = 5,
                num_sentences   = 1,
                
                tqdm    = 'auto',
                
                ** kwargs
               ):
        if not max_input_len: max_input_len = self.max_input_length
        if is_dataframe(texts):              texts = texts.to_dict('records')
        elif isinstance(texts, (str, dict)): texts = [texts]
        if tqdm == 'auto': tqdm = tqdm_progress_bar if len(texts) > 1 else lambda x: x
        
        predictions = {}
        if save:
            if directory is None: directory = self.pred_dir
            os.makedirs(directory, exist_ok = True)
            
            map_file = os.path.join(directory, filename)

            predictions = load_json(map_file, default = {})

        inputs = self.get_input(texts, ** kwargs)
        if max_input_len: inputs = [inp for inp in inputs if len(inp) < max_input_len]
        input_texts = self.decode_text(inputs)
        
        if save and not overwrite:
            indexes_to_pred = [i for i, txt in enumerate(input_texts) if txt not in predictions]
        else:
            indexes_to_pred = list(range(len(inputs)))
        # This will produce batches with less padding
        indexes_to_pred = sorted(indexes_to_pred, key = lambda idx: len(inputs[idx]), reverse = True)
        inputs = [inputs[idx] for idx in indexes_to_pred]
        
        for i in tqdm(range(0, len(inputs), batch_size)):
            out = self.compiled_infer(
                inputs[i : i + batch_size],
                max_input_len   = max_input_len,
                max_new_tokens  = max_new_tokens,
                ** kwargs
            )
            out = self.decode_output(out)

            for j, (idx, pred) in enumerate(zip(indexes_to_pred[i : i + batch_size], out)):
                if isinstance(pred, list) and len(pred) == 1: pred = pred[0]
                predictions[input_texts[idx]] = {
                    ** (texts[idx] if isinstance(texts[idx], dict) else {}),
                    'input'     : input_texts[idx],
                    'input_tokens'  : inputs[i + j],
                    'predicted'     : pred
                }

        if save: dump_json(map_file, predictions, indent = 4)

        return [predictions[txt] for txt in input_texts]
    
    @add_prompt_wrapper('qa')
    def answer(self, question, *, possible_answers = None, ** kwargs):
        return self.predict(question, ** kwargs)

    translate   = add_prompt_wrapper('translation', fn = answer)
    reformulate = add_prompt_wrapper('reformulation', fn = answer)
    summarize   = add_prompt_wrapper('summarization', fn = answer)
    extract_entities    = add_prompt_wrapper('entity_extraction', fn = answer)

    @add_prompt_wrapper('rag')
    def rag(self,
            question,
            *,
            
            k   = 5,
            reverse = True,
            vectors = None,
            retriever   = None,
            documents   = None,
            search_on_web   = None,
            search_config   = {},
            retriever_config    = {},
               
            ** kwargs
           ):
        if search_on_web is None: search_on_web = documents is None and vectors is None
        if isinstance(retriever, str):
            from models import get_pretrained
            retriever = get_pretrained(retriever)
        
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
            save_db = retriever_config.pop('save', True)
            
            vectors = retriever.predict(
                parsed, to_numpy = True, save = False, ** retriever_config
            )
            if save_db:
                self.add_vectors_database(vectors, ** retriever_config)
        
        if vectors is not None:
            if isinstance(vectors, str):
                vectors = self.get_vectors_database(name = vectors, ** kwargs)
            
            parsed = vectors.search(question, k = k, reverse = reverse)

        return self.predict(question, documents = parsed, ** kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_format' : self.output_format,
            'max_output_length' : self.max_output_length
        })
        return config

    @staticmethod
    def infer_to_str(text, score = None, indent = 0):
        _indentation = ' ' * indent
        if isinstance(text, list) and len(text) == 1:
            if score is not None: score = score[0]
            text = text[0]
        
        if not isinstance(text, (list, tuple)):
            if isinstance(text, dict): text, score = text['text'], text['score']
            return '{}Inference ({:.3f}) : {}'.format(_indentation, score, text)

        des = '{}Inference :'.format(_indentation)
        for i, txt in enumerate(text):
            if isinstance(txt, dict): txt, s = txt['text'], txt['score']
            else: s = score[i]
            des += '\n{}  #{} ({:.3f}) : {}'.format(_indentation, i + 1, s, txt)
        return des

    