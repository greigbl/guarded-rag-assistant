# Copyright 2024 DataRobot, Inc. and its affiliates.
# All rights reserved.
# DataRobot, Inc.
# This is proprietary source code of DataRobot, Inc. and its
# affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.

import json
import textwrap

import datarobot as dr
import pulumi_datarobot as datarobot

from .common.globals import GlobalGuardrailTemplateName
from .common.schema import (
    Condition,
    CustomModelArgs,
    CustomModelGuardConfigurationArgs,
    DeploymentArgs,
    GuardConditionComparator,
    Intervention,
    ModerationAction,
    RegisteredModelArgs,
    Stage,
)
from .settings_main import (
    default_prediction_server_id,
    project_name,
    runtime_environment_moderations,
)

keyword_guard_target_name = "flagged"
keyword_guard_positive_class_label = "true"
keyword_guard_negative_class_label = "false"

custom_model_args = CustomModelArgs(
    resource_name="keyword-guard-model",
    name="Keyword Guard Model",
    description="This model is designed to guard against questions about competitors",
    base_environment_id=runtime_environment_moderations.id,
    target_name=keyword_guard_target_name,
    target_type=dr.enums.TARGET_TYPE.BINARY,
    positive_class_label=keyword_guard_positive_class_label,
    negative_class_label=keyword_guard_negative_class_label,
    runtime_parameter_values=[
        datarobot.CustomModelRuntimeParameterValueArgs(
            key="blocklist",
            type="string",
            value=json.dumps(
                [
                    "dataiku",
                    "databrick",
                    "h20",
                    "aws",
                    "amazon",
                    "azure",
                    "microsoft",
                    "gcp",
                    "google",
                    "vertex\\s*ai",
                    "compet",
                ]
            ),
        ),
        datarobot.CustomModelRuntimeParameterValueArgs(
            key="prompt_feature_name",
            type="string",
            value="guardrailText",
        ),
    ],
    folder_path="deployment_keyword_guard",
)

registered_model_args = RegisteredModelArgs(
    resource_name="keyword-guard-registered-model",
    name=f"Keyword Guard Registered Model [{project_name}]",
)

deployment_args = DeploymentArgs(
    resource_name="keyword-guard-deployment",
    label="Keyword Guard Deployment",
    predictions_settings=None
    if default_prediction_server_id
    else datarobot.DeploymentPredictionsSettingsArgs(
        min_computes=0, max_computes=1, real_time=True
    ),
)

custom_model_guard_configuration_args = CustomModelGuardConfigurationArgs(
    template_name=GlobalGuardrailTemplateName.CUSTOM_DEPLOYMENT,
    name="Keyword Guard Configuration",
    stages=[Stage.PROMPT],
    intervention=Intervention(
        action=ModerationAction.BLOCK,
        condition=Condition(
            comparand=1,
            comparator=GuardConditionComparator.EQUALS,
        ).model_dump_json(),
        message=textwrap.dedent(
            """\
                I have detected you are asking about another vendor. I hear they have great products, but I think DataRobot is the best.

                For information on integrations, please check our website here:
                https://docs.datarobot.com/en/docs/more-info/how-to/index.html"""
        ),
    ),
    input_column_name="guardrailText",
    output_column_name=f"{keyword_guard_target_name}_{keyword_guard_positive_class_label}_PREDICTION",
)
