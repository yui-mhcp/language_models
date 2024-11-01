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

@keras.saving.register_keras_serializable('custom_metrics')
class EER(keras.metrics.AUC):
    def result(self):
        auc = super().result()
        
        tp_rate = K.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )
        fp_rate = K.divide_no_nan(
            self.false_positives, self.false_positives + self.true_negatives
        )
        fn_rate = 1 - tp_rate
        
        diff    = K.abs(fp_rate - fn_rate)
        min_index   = K.argmin(diff)
        
        eer = (fp_rate[min_index] + fn_rate[min_index]) / 2.
        return {'eer' : eer, 'auc' : auc}
