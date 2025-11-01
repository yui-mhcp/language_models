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

from abc import ABC, abstractmethod

class ConvItemSelector(ABC):
    """ Abstract item selector class that selects items from a list """
    def __init__(self, directory, tokenizer, ** _):
        self.directory  = directory
        self.tokenizer  = tokenizer
    
    def __repr__(self):
        return '<{} {}>'.format(
            self.__class__.__name__,
            ' '.join('{}={}'.format(k, v) for k, v in self.config.items())
        )
    
    def __call__(self, query, items, *, conv = None, max_length = None, ** kwargs):
        return self.select(conv, items, max_length = max_length, ** kwargs)

    @abstractmethod
    def select(self, query, items, *, conv = None, max_length = None, ** kwargs):
        """ Return a `list` of item from the given `items` """
    
    def get_config(self):
        return {}
