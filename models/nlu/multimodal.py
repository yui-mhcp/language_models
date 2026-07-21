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
import uuid
import numpy as np

from threading import Lock

from utils import load_json, dump_json

class MultimodalMixin:
    """
        Generic multimodal handling for a `BaseLLM` : each modality `k` (image / audio / video)
        found in `data` is prepared through `self.prepare_{k}` when the model defines the
        associated `{k}_token`, then inserted in the token sequence (`insert_multimodal_tokens`).

        This is the **generic (keras) path** : runtime-specific handling (e.g. the TRT-LLM
        `prompt_table`) is implemented by the model overrides (see `Mistral3`).
    """
    def insert_multimodal_tokens(self, tokens, multimodal_inputs):
        raise NotImplementedError('{} does not support multimodal token insertion'.format(
            self.__class__.__name__
        ))

    def prepare_multimodal_data(self, tokens, *, directory = None, ** data):
        multimodal_inputs = {
            k : getattr(self, 'prepare_{}'.format(k))(v)
            for k, v in data.items()
            if getattr(self, '{}_token'.format(k), None) is not None
        }

        if multimodal_inputs:
            tokens = self.insert_multimodal_tokens(tokens, multimodal_inputs)

        return tokens, multimodal_inputs

class ImageFeatureCache:
    """
        Thread-safe, optionally persistent, cache mapping an image filename to its
        `(infos, features)` couple (`infos` at least holds `height` / `width`).

        Arguments :
            - encode_fn : `callable(image) -> (infos, features)` performing the effective
              vision encoding (`image` is the raw input : filename / array / ...)

        When a `directory` is provided to `get`, the entries are persisted :
            {directory}/features/
                map.json    # {filename : infos} mapping
                {id}.npy    # the features of a given image (unique `id`)

        The whole `get` runs under a mutex : concurrent requests on the same image encode it
        once, and the vision engine (a TensorRT execution context, which is **not**
        thread-safe) is never called concurrently.
    """
    def __init__(self, encode_fn):
        self._encode_fn = encode_fn

        self._mutex     = Lock()
        self._infos     = {}    # directory -> {filename : infos}
        self._features  = {}    # filename -> features

    @staticmethod
    def _feature_dir(directory):
        return os.path.join(directory, 'features')

    @classmethod
    def _map_file(cls, directory):
        return os.path.join(cls._feature_dir(directory), 'map.json')

    def _get_directory_cache(self, directory):
        if directory not in self._infos:
            cache = {}
            if directory:
                map_file = self._map_file(directory)
                if os.path.exists(map_file):
                    cache = load_json(map_file, default = {})
            self._infos[directory] = cache

        return self._infos[directory]

    def get(self, filename, *, directory = None, image = None):
        """
            Returns the `(infos, features)` couple for `filename`, computing (and possibly
            persisting) it on cache miss. `image` overrides the image data to encode (the
            entry is still keyed by `filename`).
        """
        with self._mutex:
            cache = self._get_directory_cache(directory)

            if isinstance(filename, str) and filename in cache:
                if filename in self._features:
                    return cache[filename], self._features[filename]

                feat_file = cache[filename].get('features', None)
                if feat_file and os.path.exists(feat_file):
                    self._features[filename] = np.load(feat_file)
                    return cache[filename], self._features[filename]
                # persisted entry without a (valid) feature file : recompute it

            return self._compute(filename, image, directory, cache)

    def _compute(self, filename, image, directory, cache):
        infos, features = self._encode_fn(filename if image is None else image)

        if isinstance(filename, str):
            cache[filename] = infos
            self._features[filename] = features

            if directory:
                feat_dir  = self._feature_dir(directory)
                os.makedirs(feat_dir, exist_ok = True)

                feat_file = os.path.join(feat_dir, '{}.npy'.format(uuid.uuid4().hex))
                infos['features'] = feat_file

                dump_json(self._map_file(directory), cache)
                np.save(feat_file, _to_numpy(features))

        return infos, features

def _to_numpy(features):
    if hasattr(features, 'device'):     # torch tensor, possibly bfloat16 / on GPU
        import torch
        features = features.to(dtype = torch.float32, device = torch.device('cpu'))

    from utils.keras import ops
    return ops.convert_to_numpy(features)
