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

import logging

from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)

class Runtime(metaclass = ABCMeta):
    _engines = {}
    
    def __init__(self, path, *, engine = None, ** kwargs):
        if engine is None:
            if path not in self._engines:
                self._engines[path] = self.load_engine(path, ** kwargs)
            engine = self._engines[path]
        
        self.path   = path
        self.engine = engine
    
    def __repr__(self):
        return '<{} path={}>'.format(self.__class__.__name__, self.path)
    
    @abstractmethod
    def __call__(self, * args, ** kwargs):
        """ Performs custom runtime inference """
    
    @staticmethod
    @abstractmethod
    def load_engine(path, ** kwargs):
        """ Loads the custom runtime engine """