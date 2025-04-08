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

import os
import pathlib
import sys

import pulumi
import pulumi_datarobot as datarobot

sys.path.append("..")

from docsassist.deployments import (
    app_env_name,
    rag_deployment_env_name,
)
from docsassist.i18n import LocaleSettings
from docsassist.schema import ApplicationType, RAGType
from infra import (
    settings_app_infra,
    settings_generative,
    settings_keyword_guard,
    settings_main,
)
from infra.common.feature_flags import check_feature_flags
from infra.common.papermill import run_notebook
from infra.common.urls import get_deployment_url
from infra.components.custom_model_deployment import CustomModelDeployment
from infra.components.dr_llm_credential import (
    get_credential_runtime_parameter_values,
    get_credentials,
)
from infra.components.rag_custom_model import RAGCustomModel
from infra.settings_global_guardrails import global_guardrails

LocaleSettings().setup_locale()

check_feature_flags(pathlib.Path("feature_flag_requirements.yaml"))

if "DATAROBOT_DEFAULT_USE_CASE" in os.environ:
    use_case_id = os.environ["DATAROBOT_DEFAULT_USE_CASE"]
    pulumi.info(f"Using existing use case '{use_case_id}'")
    use_case = datarobot.UseCase.get(
        id=use_case_id,
        resource_name="Guarded RAG Use Case [PRE-EXISTING]",
    )
else:
    use_case = datarobot.UseCase(**settings_main.use_case_args)

if settings_main.default_prediction_server_id is None:
    prediction_environment = datarobot.PredictionEnvironment(
        **settings_main.prediction_environment_args,
    )
else:
    prediction_environment = datarobot.PredictionEnvironment.get(
        "Guarded RAG Prediction Environment [PRE-EXISTING]",
        settings_main.default_prediction_server_id,
    )

credentials = get_credentials(settings_generative.LLM)

credential_runtime_parameter_values = get_credential_runtime_parameter_values(
    credentials=credentials
)


keyword_guard_deployment = CustomModelDeployment(
    resource_name=f"My Keyword Guard [{settings_main.project_name}]",
    custom_model_args=settings_keyword_guard.custom_model_args,
    registered_model_args=settings_keyword_guard.registered_model_args,
    prediction_environment=prediction_environment,
    deployment_args=settings_keyword_guard.deployment_args,
)

global_guard_deployments = [
    datarobot.Deployment(
        registered_model_version_id=datarobot.get_global_model(
            name=guard.registered_model_name,
        ).version_id,
        prediction_environment_id=prediction_environment.id,
        use_case_ids=[use_case.id],
        **guard.deployment_args.model_dump(),
    )
    for guard in global_guardrails
]

all_guard_deployments = [keyword_guard_deployment] + global_guard_deployments

all_guardrails_configs = [
    settings_keyword_guard.custom_model_guard_configuration_args
] + [guard.custom_model_guard_configuration_args for guard in global_guardrails]


guard_configurations = [
    datarobot.CustomModelGuardConfigurationArgs(
        deployment_id=deployment.id,
        **guard_config_args.model_dump(mode="json", exclude_none=True),
    )
    for deployment, guard_config_args in zip(
        all_guard_deployments,
        all_guardrails_configs,
    )
]

if settings_main.core.rag_type == RAGType.DR:
    rag_custom_model = RAGCustomModel(
        resource_name=f"Guarded RAG Prep [{settings_main.project_name}]",
        use_case=use_case,
        dataset_args=settings_generative.dataset_args,
        playground_args=settings_generative.playground_args,
        vector_database_args=settings_generative.vector_database_args,
        llm_blueprint_args=settings_generative.llm_blueprint_args,
        runtime_parameter_values=credential_runtime_parameter_values,
        guard_configurations=guard_configurations,
        custom_model_args=settings_generative.custom_model_args,
    )
elif settings_main.core.rag_type == RAGType.DIY:
    if not all(
        [
            path.exists()
            for path in settings_generative.diy_rag_nb_output.model_dump().values()
        ]
    ):
        pulumi.info("Executing doc chunking + vdb building notebook...")
        run_notebook(settings_generative.diy_rag_nb)
    else:
        pulumi.info(
            f"Using existing outputs from build_rag.ipynb in '{settings_generative.diy_rag_deployment_path}'"
        )

    rag_custom_model = datarobot.CustomModel(  # type: ignore[assignment]
        files=settings_generative.get_diy_rag_files(
            runtime_parameter_values=credential_runtime_parameter_values,
        ),
        runtime_parameter_values=credential_runtime_parameter_values,
        guard_configurations=guard_configurations,
        use_case_ids=[use_case.id],
        **settings_generative.custom_model_args.model_dump(
            mode="json", exclude_none=True
        ),
    )
else:
    raise NotImplementedError(f"Unknown RAG type: {settings_main.core.rag_type}")

rag_deployment = CustomModelDeployment(
    resource_name=f"Guarded RAG Deploy [{settings_main.project_name}]",
    custom_model_version_id=rag_custom_model.version_id,
    registered_model_args=settings_generative.registered_model_args,
    prediction_environment=prediction_environment,
    deployment_args=settings_generative.deployment_args,
    use_case_ids=[use_case.id],
)

app_runtime_parameters = [
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key=rag_deployment_env_name, type="deployment", value=rag_deployment.id
    ),
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key="APP_LOCALE", type="string", value=LocaleSettings().app_locale
    ),
]

if settings_main.core.application_type == ApplicationType.DIY:
    application_source = datarobot.ApplicationSource(
        runtime_parameter_values=app_runtime_parameters,
        **settings_app_infra.app_source_args,
    )
    qa_application = datarobot.CustomApplication(
        resource_name=settings_app_infra.app_resource_name,
        source_version_id=application_source.version_id,
        use_case_ids=[use_case.id],
    )
elif settings_main.core.application_type == ApplicationType.DR:
    qa_application = datarobot.QaApplication(  # type: ignore[assignment]
        resource_name=settings_app_infra.app_resource_name,
        name=f"Guarded RAG Assistant [{settings_main.project_name}]",
        deployment_id=rag_deployment.deployment_id,
        opts=pulumi.ResourceOptions(delete_before_replace=True),
    )
else:
    raise NotImplementedError(
        f"Unknown application type: {settings_main.core.application_type}"
    )

qa_application.id.apply(settings_app_infra.ensure_app_settings)


pulumi.export(rag_deployment_env_name, rag_deployment.id)
pulumi.export(app_env_name, qa_application.id)
for deployment, config in zip(global_guard_deployments, global_guardrails):
    pulumi.export(
        config.deployment_args.resource_name,
        deployment.id.apply(get_deployment_url),
    )

pulumi.export(
    settings_generative.deployment_args.resource_name,
    rag_deployment.id.apply(get_deployment_url),
)
pulumi.export(
    settings_app_infra.app_resource_name,
    qa_application.application_url,
)
