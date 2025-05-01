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
import requests

from datetime import datetime

from .tool import Tool
from ..prompts import prompt_docstring

base_url = "http://api.openweathermap.org/data/2.5/weather"

@prompt_docstring(
    en = "Get weather information from the OpenWeatherMap API",
    fr = "Récupère les informations météos de l'API OpenWeatherMap"
)
def get_weather(location : str, date : datetime = None):
    params = {"q" : location, "appid" : os.environ['OPENWEATHERMAP_API_KEY'], "units" : "metric"}
    if date: params["dt"] = date.timestamp()
    
    response = requests.get(base_url, params = params)
    if response.status_code == 200:
        res = response.json()
        return {
            'description'   : res['weather'][0]['description'],
            'temperature'   : {k : v for k, v in res['main'].items() if 'temp' in k},
            'humidity'  : res['main']['humidity'],
            'wind'      : res['wind'],
            'clouds'    : '{} %'.format(res['clouds']['all']),
            'sunset'    : datetime.fromtimestamp(res['sys']['sunset']).strftime("%Hh %Mmin"),
            'sunrize'   : datetime.fromtimestamp(res['sys']['sunrise']).strftime("%Hh %Mmin")
        }
    else:
        return response.json()['message']

OpenWeatherMapTool = Tool.from_function(get_weather)