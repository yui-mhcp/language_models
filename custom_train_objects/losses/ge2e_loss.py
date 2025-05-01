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

import keras
import keras.ops as K

@keras.saving.register_keras_serializable('custom_loss')
class GE2ELoss(keras.losses.Loss):
    def __init__(self,
                 mode = 'softmax',
                 distance_metric = 'cosine',
                 
                 init_w = 1.,
                 init_b = 0.,
                 
                 ** kwargs
                ):
        if not hasattr(self, '{}_loss'.format(mode)):
            raise ValueError('Unknown `mode` : {}'.format(mode))
        
        super().__init__(** kwargs)
        self.mode   = mode
        self.distance_metric    = distance_metric
        
        self.w  = abs(init_w)
        self.b  = init_b
        
        self.loss_fn    = getattr(self, '{}_loss'.format(mode))
    
    def init_variables(self, model):
        if any(v.name == 'ge2e_loss_w' for v in model.variables):
            w = [v for v in model.variables if v.name == 'ge2e_loss_w'][0]
            b = [v for v in model.variables if v.name == 'ge2e_loss_b'][0]
            self.w = w
            self.b = b
        else:
            model._tracker.unlock()
            self.w = model.add_variable(
                shape   = (),
                name    = 'ge2e_loss_w',
                initializer = lambda shape, dtype: K.convert_to_tensor(self.w, dtype)
            )
            self.b = model.add_variable(
                shape   = (),
                name    = 'ge2e_loss_b',
                initializer = lambda shape, dtype: K.convert_to_tensor(self.b, dtype)
            )
            model._tracker.lock()

    def softmax_loss(self, ids, similarity_matrix):
        return K.sparse_categorical_crossentropy(
            ids, similarity_matrix, from_logits = True
        )
    
    def contrast_loss(self, ids, similarity_matrix):
        ids = K.one_hot(ids, depth = K.shape(similarity_matrix)[1])
        return K.binary_crossentropy(
            ids, similarity_matrix, from_logits = True
        )
    
    def call(self, y_true, y_pred):
        """
            Implementation of the GE2E loss function with centroid exclusion
            
            Arguments : 
                - y_true : the ids for each embedding with shape `(n_labels, n_samples)`
                - y_pred : embeddings matrix
                    shape `(n_labels * n_samples, embedding_dim)`
            Return :
                - loss  : the loss value for each sample (depends on `self.mode`)
        """
        n_labels    = K.shape(y_true)[0]
        n_samples   = K.shape(y_true)[1]
        
        # shape == (n_labels, n_samples, embedded_dim)
        reshaped    = K.reshape(y_pred, [n_labels, n_samples, K.shape(y_pred)[1]])
        
        sim_matrix = self.similarity_matrix(reshaped, self.distance_metric)
        sim_matrix = K.reshape(sim_matrix, [K.shape(y_pred)[0], n_labels])
        sim_matrix = sim_matrix * self.w + self.b

        target  = K.repeat(K.arange(n_labels, dtype = 'int32'), n_samples, axis = 0)
        return self.loss_fn(target, sim_matrix)

    def get_config(self):
        config = super().get_config()
        config.update({
            'mode'  : self.mode,
            'init_w'    : float(K.convert_to_numpy(self.w)),
            'init_b'    : float(K.convert_to_numpy(self.b)),
            'distance_metric'   : self.distance_metric
        })
        return config

    @staticmethod
    def similarity_matrix(embeddings, distance_metric):
        """
            Implementation of the similarity matrix with exclusive centroids (equation 9)

            Arguments : 
                - embeddings    : the embeddings
                    shape = `(n_labels, n_samples_per_label, embedding_dim)`
            Returns :
                - similarity_matrix : the similarity score between each sample and each centroid (with exclusion)
                    shape = `(n_labels * n_samples_per_label, n_labels)`
        """
        n_labels    = K.shape(embeddings)[0]
        n_samples   = K.shape(embeddings)[1]

        # Shape == (n_labels, embedded_dim)
        centroids_incl = K.mean(embeddings, axis = 1)

        # Shape == (n_labels, n_samples, embedded_dim) == embeddings.shape
        centroids_excl = K.sum(embeddings, axis = 1, keepdims = True) - embeddings
        centroids_excl = centroids_excl / K.cast(n_samples - 1, embeddings.dtype)

        # Compute mask (shape = (nb_speakers, nb_utterances, nb_speakers, 1))
        if keras.backend.backend() == 'tensorflow':
            import tensorflow as tf
            mask = tf.eye(n_labels, dtype = 'bool')
        else:
            mask = K.eye(n_labels, dtype = 'bool')
        mask = K.repeat(mask, n_samples, axis = 0)
        mask = K.reshape(mask, [n_labels, n_samples, n_labels, 1])

        # Shape == (n_labels, n_samples, n_labels, embedding_dim)
        centroids   = K.where(
            mask, centroids_excl[:, :, None, :], centroids_incl[None, None, :, :]
        )
        embeddings  = embeddings[:, :, None, :]
        if distance_metric == 'euclidian':
            xx = _einsum_matmul(embeddings, embeddings)
            yy = _einsum_matmul(centroids, centroids)
            xy = _einsum_matmul(embeddings, centroids)
            return - K.sqrt(xx - 2 * xy + yy)
        elif distance_metric == 'cosine':
            return _einsum_matmul(
                K.divide_no_nan(embeddings, K.norm(embeddings, axis = -1, keepdims = True)),
                K.divide_no_nan(centroids, K.norm(centroids, axis = -1, keepdims = True))
            )
        elif distance_metric == 'dp':
            return _einsum_matmul(label_embeddings, centroids)
        elif distance_metric == 'manhattan':
            return - K.sum(K.abs(embeddings - centroids), axis = -1)

def _einsum_matmul(x, y):
    return K.einsum('...i, ...i -> ...', x, y)