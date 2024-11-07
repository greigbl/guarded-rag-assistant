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

import pulumi_datarobot as datarobot

from docsassist.schema import Grade

from .common.globals import GlobalRuntimeEnvironment
from .common.schema import (
    CustomModelArgs,
    DeploymentArgs,
    RegisteredModelArgs,
)
from .settings_main import default_prediction_server_id, project_name

custom_model_args = CustomModelArgs(
    resource_name=f"Grading Custom Model [{project_name}]",
    files=[
        ("deployment_grading/custom.py", "custom.py"),
        ("deployment_grading/requirements.txt", "requirements.txt"),
        ("docsassist/schema.py", "docsassist/schema.py"),
    ],
    base_environment_id=GlobalRuntimeEnvironment.PYTHON_311_GENAI.value.id,
    target_type="Multiclass",
    target_name="grade",
    class_labels=list(get_args(Grade)),
)

registered_model_args = RegisteredModelArgs(
    resource_name=f"Grading Registered Model [{project_name}]",
)


deployment_args = DeploymentArgs(
    resource_name=f"Grading Deployment [{project_name}]",
    label=f"Grading Deployment [{project_name}]",
    predictions_settings=(
        None
        if default_prediction_server_id
        else datarobot.DeploymentPredictionsSettingsArgs(
            min_computes=0, max_computes=1, real_time=True
        )
    ),
    predictions_data_collection_settings=datarobot.DeploymentPredictionsDataCollectionSettingsArgs(
        enabled=True
    ),
)
