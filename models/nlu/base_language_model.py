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

from utils import partial, is_dataframe, normalize_keys
from utils.keras_utils import ops
from loggers import timer, time_logger
from utils.text import get_encoder
from models.interfaces.base_text_model import BaseTextModel
from custom_architectures.transformers_arch import get_pretrained_transformer

_alternative_keys   = {
    'text'      : ('context', 'content', 'paragraph'),
    'answer'    : ('answers', ),
    'context'   : ('paragraph', 'text', 'contexts', 'paragraphs', 'texts'),
    'title'     : ('titles', )
}


class BaseLanguageModel(BaseTextModel):
    _default_pretrained_model   = None
    
    input_signature = BaseTextModel.text_signature
    decode_output   = BaseTextModel.decode_text
    
    def __init__(self, lang, input_format = None, max_input_length = None, ** kwargs):
        pretrained = kwargs.pop('pretrained', self._default_pretrained_model)
        if pretrained and 'text_encoder' not in kwargs:
            kwargs['text_encoder'] = get_encoder(lang = None, text_encoder = pretrained)

        self._init_text(lang = lang, ** kwargs)
        
        self.input_format   = input_format
        self.max_input_length   = max_input_length
        
        if isinstance(pretrained, str): kwargs.setdefault('pretrained_name', pretrained)
        super().__init__(pretrained = pretrained, ** kwargs)
        
        if hasattr(self.model, 'set_tokens'): self.model.set_tokens(** self.model_tokens)
    
    def build(self, *, model = None, pretrained = None, ** kwargs):
        if model is not None: return super().build(model = model)
        
        if pretrained is not None:
            super().build(
                model = get_pretrained_transformer(pretrained, ** kwargs)
            )
        else:
            super().build(** kwargs)
    
    @property
    def encoder(self):
        return self.model if not self.is_encoder_decoder else getattr(self.model, 'encoder', None)
    
    @property
    def decoder(self):
        return getattr(self.model, 'decoder', None)
    
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
        
        return self.prepare_text(
            normalize_keys(data, _alternative_keys), format = input_format, ** kwargs
        )
    
    def filter_input(self, inputs):
        """ Check `is_valid_tokens` for information """
        if self.is_encoder_decoder:     inputs = inputs[0]
        
        if not self.max_input_length:
            return ops.all(ops.shape(inputs) > 0)
        
        return ops.logical_and(
            ops.all(ops.shape(inputs) > 0),
            ops.shape(inputs)[-1] <= self.max_input_length
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            ** self.get_config_text(),
            'input_format'  : self.input_format,
            'max_input_length'  : self.max_input_length
        })
        return config
        
