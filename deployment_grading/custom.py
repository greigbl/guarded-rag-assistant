# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import get_args

from docsassist.schema import Grade

CLASS_LABELS = list(get_args(Grade))


# dummy load model function to overwrite the default
def load_model(code_dir):
    return ""


# equal prediction on each class
def score(data, model, **kwargs):
    equal_weight = 1.0 / len(CLASS_LABELS)
    for col in CLASS_LABELS:
        data[col] = equal_weight
    return data[CLASS_LABELS]
