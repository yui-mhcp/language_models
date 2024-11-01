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

import keras
import keras.ops as K

from ..losses.ge2e_loss import GE2ELoss

@keras.saving.register_keras_serializable('custom_metrics')
class GE2EMetric(keras.metrics.Metric):
    def __init__(self, mode = 'softmax', distance_metric = 'cosine', ** kwargs):
        assert mode in ('softmax', 'contrast')
        
        super().__init__(** kwargs)
        
        self.mode   = mode
        self.distance_metric    = distance_metric
        
        if mode == 'softmax':
            self.metric     = keras.metrics.SparseCategoricalAccuracy()
            self.format_fn  = self.softmax_format
        else:
            from .equal_error_rate import EER
            
            self.metric     = EER()
            self.format_fn  = self.contrast_format
    
    def reset_state(self, * args, ** kwargs):
        self.metric.reset_state(* args, ** kwargs)
    
    def softmax_format(self, ids, similarity_matrix):
        return ids, similarity_matrix
    
    def contrast_format(self, idx, similarity_matrix):
        return K.one_hot(ids, depth = K.shape(similarity_matrix)[1]), similarity_matrix
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        n_labels    = K.shape(y_true)[0]
        n_samples   = K.shape(y_true)[1]
        
        # shape == (n_labels, n_samples, embedded_dim)
        reshaped    = K.reshape(y_pred, [n_labels, n_samples, K.shape(y_pred)[1]])
        
        sim_matrix = GE2ELoss.similarity_matrix(reshaped, self.distance_metric)
        sim_matrix = K.reshape(sim_matrix, [K.shape(y_pred)[0], n_labels])

        target  = K.repeat(K.arange(n_labels, dtype = 'int32'), n_samples, axis = 0)
        
        target, pred = self.format_fn(target, sim_matrix)

        return self.metric.update_state(target, pred, sample_weight)
        
    def result(self):
        return self.metric.result()
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'mode'  : self.mode,
            'distance_metric'   : self.distance_metric
        })
        return config
