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
import numpy as np

from threading import Lock

from loggers import timer
from utils import load_json, dump_json
from utils.keras import ops
from .text_generator import TextGenerator
from ..interfaces.base_image_model import BaseImageModel

class Mistral3(BaseImageModel, TextGenerator):
    def __init__(self,
                 * args,
                 
                 patch_size = None,
                 spatial_merge_size = None,
                 
                 kv_cache_enable_block_reuse    = True,
                 
                 init_keras = True,
                 
                 ** kwargs
                ):
        if patch_size is None and kwargs.get('pretrained', None):
            from transformers import AutoConfig, AutoProcessor
            
            hf_config = AutoConfig.from_pretrained(kwargs['pretrained'])
            processor = AutoProcessor.from_pretrained(kwargs['pretrained']).image_processor
            
            kwargs['input_size'] = (None, None, 3)
            patch_size = processor.patch_size
            spatial_merge_size  = hf_config.spatial_merge_size
            
            kwargs.setdefault('resize_kwargs', {}).update({
                'multiples' : patch_size * spatial_merge_size,
                'max_shape' : 1540,
                'interpolation' : 'bicubic',
                'preserve_aspect_ratio' : True
            })
            kwargs['image_normalization']   = {
                'means' : processor.image_mean,
                'std'   : processor.image_std
            }
        
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        
        self._init_image(** kwargs)
        
        kwargs['kv_cache_enable_block_reuse'] = kv_cache_enable_block_reuse
        super().__init__(* args, ** kwargs)
        
        self.mutex  = Lock()
        self.image_to_infos = {}
        self.image_to_features  = {}
        self.image_to_extra_token   = {}
        self.last_extra_token_index = self.vocab_size
        self.kv_cache_enable_block_reuse = kv_cache_enable_block_reuse
        
        if init_keras: ops.zeros(())
    
    @property
    def image_dtype(self):
        return 'float32' if self.runtime != 'trt_llm' else self.model.multimodal_engine.dtypes[0]
    
    @property
    def image_token(self):
        return '[IMG]'
    
    @property
    def image_break_token(self):
        return '[IMG_BREAK]'
    
    @property
    def image_end_token(self):
        return '[IMG_END]'
    
    @property
    def image_token_idx(self):
        return self.tokenizer[self.image_token]
    
    @property
    def image_end_token_idx(self):
        return self.tokenizer[self.image_end_token]

    @property
    def image_break_token_idx(self):
        return self.tokenizer[self.image_break_token]

    def get_image_features(self, filename, directory = None):
        if directory:
            feat_dir  = os.path.join(directory, 'features')
            map_file  = os.path.join(feat_dir, 'map.json')
            if directory not in self.image_to_infos:
                if os.path.exists(map_file):
                    self.image_to_infos[directory] = cache = load_json(map_file, default = {})
                else:
                    os.makedirs(feat_dir, exist_ok = True)
                    self.image_to_infos[directory] = cache = {}
            else:
                cache = self.image_to_infos[directory]
        else:
            cache = self.image_to_infos.setdefault(directory, {})
        
        if isinstance(filename, str) and filename in cache:
            infos = cache[filename]
            if filename not in self.image_to_features:
                self.image_to_features[filename] = np.load(infos['features'])
            
            return infos, self.image_to_features[filename]
        
        import torch

        image   = self.prepare_image(filename)
        mask    = torch.zeros(
            1,
            image.shape[0] // self.patch_size,
            image.shape[1] // self.patch_size,
            dtype  = getattr(torch, self.model.multimodal_engine.dtypes[1]),
            device = torch.device('cuda')
        )
        features    = self.model.encode_multimodal_data(image = image, attention_mask = mask)
        
        infos = {'height' : image.shape[0], 'width' : image.shape[1]}
        
        if isinstance(filename, str):
            cache[filename] = infos
            self.image_to_features[filename] = features
            
            if directory:
                feat_file = os.path.join(feat_dir, '{}.npy'.format(time.time()))
                
                infos['features'] = feat_file
                
                dump_json(map_file, infos)
                np.save(feat_file, ops.convert_to_numpy(features.to(
                    dtype = torch.float32, device = torch.device('cpu')
                )))
        
        return infos, features
    
    def get_extra_tokens(self, filename, start_idx, patch_h, patch_w):
        if not self.kv_cache_enable_block_reuse:
            if start_idx is None: start_idx = self.vocab_size
            last_idx = start_idx + patch_h * patch_w
            return list(range(start_idx, last_idx)), last_idx
        elif isinstance(filename, str) and filename in self.image_to_extra_token:
            return self.image_to_extra_token[filename], start_idx
        
        with self.mutex:
            if isinstance(filename, str) and filename in self.image_to_extra_token:
                return self.image_to_extra_token[filename], start_idx
            
            if start_idx is None: start_idx = self.last_extra_token_index
            last_idx  = start_idx + patch_h * patch_w
            self.last_extra_token_index = last_idx
            extra_tokens = list(range(start_idx, last_idx))
            
            if isinstance(filename, str):
                self.image_to_extra_token[filename] = list(range(start_idx, last_idx))
        
        return extra_tokens, last_idx
    
    @timer
    def prepare_multimodal_data(self, tokens, *, directory = None, ** data):
        if not data.get('image', None):
            return tokens, {}
        elif self.runtime != 'trt_llm':
            return super().prepare_multimodal_data(tokens, ** data)
        else:
            import torch
            
            indexes = np.where(tokens == self.image_token_idx)[0]
            images  = data['image']

            if len(indexes) != len(images):
                raise RuntimeError('There is {} image tokens for {} images'.format(
                    len(indexes), len(images)
                ))

            last_idx    = 0
            mm_tokens   = []
            features    = []
            start_extra_token   = None
            image_extra_tokens  = []
            for i, (filename, idx) in enumerate(zip(images, indexes)):
                if isinstance(filename, dict): filename = filename['image']
                infos, image_features = self.get_image_features(filename, directory = directory)
                
                patch_h = infos['height'] // (self.patch_size * self.spatial_merge_size)
                patch_w = infos['width'] // (self.patch_size * self.spatial_merge_size)

                row = [self.image_token_idx] * patch_w + [self.image_break_token_idx]
                image_tokens = row * patch_h

                mm_tokens.append(tokens[last_idx : idx])
                mm_tokens.append(image_tokens[:-1] + [self.image_end_token_idx])
                features.append(image_features)
                
                extra_tokens, start_extra_token = self.get_extra_tokens(
                    filename, start_extra_token, patch_h, patch_w
                )
                image_extra_tokens.extend(extra_tokens)

                last_idx = idx + 1

            mm_tokens.append(tokens[last_idx :])
            mm_tokens = np.concatenate(mm_tokens, axis = 0)
            extra_tokens = mm_tokens.copy()
            
            mask = mm_tokens == self.image_token_idx
            mm_tokens[mask] = np.arange(self.vocab_size, self.vocab_size + len(image_extra_tokens))
            
            dtype = getattr(torch, self.model.multimodal_engine.dtypes[0])
            
            features = [ops.convert_to_torch_tensor(f, dtype = dtype) for f in features]
            features = torch.concat(features, dim = 1) if len(features) > 1 else features[0]
            multimodal_inputs = {'prompt_table' : features}
            if self.kv_cache_enable_block_reuse:
                extra_tokens[mask] = np.array(image_extra_tokens)
                multimodal_inputs['input_token_extra_ids'] = [extra_tokens.tolist()]
            
            return mm_tokens, multimodal_inputs
    
    def insert_multimodal_tokens(self, tokens, multimodal_inputs):
        indexes = np.where(tokens == self.image_token_idx)[0]
        images  = multimodal_inputs['image']
        
        if len(indexes) != len(images):
            raise RuntimeError('There is {} image tokens for {} images'.format(
                len(indexes), len(images)
            ))
        
        last_idx  = 0
        mm_tokens = []
        for i, idx in enumerate(indexes):
            patch_h = images[i].shape[-3] // (self.patch_size * self.spatial_merge_size)
            patch_w = images[i].shape[-2] // (self.patch_size * self.spatial_merge_size)
            
            row = [self.image_token_idx] * patch_w + [self.image_break_token_idx]
            image_tokens = row * patch_h
            
            mm_tokens.append(tokens[last_idx : idx])
            mm_tokens.append(image_tokens[:-1] + [self.image_end_token_idx])
            
            last_idx = idx + 1
        
        mm_tokens.append(tokens[last_idx :])
        return np.concatenate(mm_tokens, axis = 0)

    def get_config(self):
        config = super().get_config()
        config.update(self.get_config_image())
        config.update({
            'patch_size'    : self.patch_size,
            'spatial_merge_size'    : self.spatial_merge_size
        })
        return config
    