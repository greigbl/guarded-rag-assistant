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

import pulumi
import pulumi_datarobot as datarobot

from docsassist.deployments import (
    app_env_name,
    grading_deployment_env_name,
    rag_deployment_env_name,
)
from docsassist.i18n import LocaleSettings
from infra import (
    settings_app_infra,
    settings_grader,
    settings_keyword_guard,
    settings_main,
    settings_rag,
)
from infra.common.feature_flags import check_feature_flags
from infra.common.papermill import run_notebook
from infra.common.urls import get_deployment_url
from infra.components.custom_model_deployment import CustomModelDeployment
from infra.components.dr_credential import DRCredential
from infra.components.rag_custom_model import RAGCustomModel
from infra.settings_global_guardrails import global_guardrails
from infra.settings_llm_credential import credential, credential_args

LocaleSettings().setup_locale()

check_feature_flags(pathlib.Path("infra/feature_flag_requirements.yaml"))

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

llm_credential = DRCredential(
    resource_name=f"Generic LLM Credential [{settings_main.project_name}]",
    credential=credential,
    credential_args=credential_args,
)

keyword_guard_deployment = CustomModelDeployment(
    resource_name=f"Keyword Guard [{settings_main.project_name}]",
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

if settings_main.core.rag_type == settings_main.RAGType.DR:
    rag_custom_model = RAGCustomModel(
        resource_name=f"Guarded RAG Prep [{settings_main.project_name}]",
        use_case=use_case,
        dataset_args=settings_rag.dataset_args,
        playground_args=settings_rag.playground_args,
        vector_database_args=settings_rag.vector_database_args,
        llm_blueprint_args=settings_rag.llm_blueprint_args,
        runtime_parameter_values=llm_credential.runtime_parameter_values,
        guard_configurations=guard_configurations,
        custom_model_args=settings_rag.custom_model_args,
    )
elif settings_main.core.rag_type == settings_main.RAGType.DIY:
    if not all(
        [path.exists() for path in settings_rag.diy_rag_nb_output.model_dump().values()]
    ):
        pulumi.info("Executing doc chunking + vdb building notebook...")
        run_notebook(settings_rag.diy_rag_nb)
    else:
        pulumi.info(
            f"Using existing outputs from build_rag.ipynb in '{settings_rag.diy_rag_deployment_path}'"
        )

    rag_custom_model = datarobot.CustomModel(  # type: ignore[assignment]
        files=settings_rag.get_diy_rag_files(
            runtime_parameter_values=llm_credential.runtime_parameter_values,
        ),
        runtime_parameter_values=llm_credential.runtime_parameter_values,
        guard_configurations=guard_configurations,
        **settings_rag.custom_model_args.model_dump(mode="json", exclude_none=True),
    )
else:
    raise NotImplementedError(f"Unknown RAG type: {settings_main.core.rag_type}")

rag_deployment = CustomModelDeployment(
    resource_name=f"Guarded RAG Deploy [{settings_main.project_name}]",
    custom_model_version_id=rag_custom_model.version_id,
    registered_model_args=settings_rag.registered_model_args,
    prediction_environment=prediction_environment,
    deployment_args=settings_rag.deployment_args,
)

grading_deployment = CustomModelDeployment(
    resource_name=f"Grading [{settings_main.project_name}]",
    custom_model_args=settings_grader.custom_model_args,
    registered_model_args=settings_grader.registered_model_args,
    prediction_environment=prediction_environment,
    deployment_args=settings_grader.deployment_args,
)

app_runtime_parameters = [
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key=rag_deployment_env_name, type="deployment", value=rag_deployment.id
    ),
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key=grading_deployment_env_name, type="deployment", value=grading_deployment.id
    ),
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key="APP_LOCALE", type="string", value=LocaleSettings().app_locale
    ),
]

if settings_main.core.application_type == settings_main.ApplicationType.DIY:
    application_source = datarobot.ApplicationSource(
        runtime_parameter_values=app_runtime_parameters,
        **settings_app_infra.app_source_args,
    )
    qa_application = datarobot.CustomApplication(
        resource_name=settings_app_infra.app_resource_name,
        source_version_id=application_source.version_id,
    )
elif settings_main.core.application_type == settings_main.ApplicationType.DR:
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


pulumi.export(grading_deployment_env_name, grading_deployment.id)
pulumi.export(rag_deployment_env_name, rag_deployment.id)
pulumi.export(app_env_name, qa_application.id)
for deployment, config in zip(global_guard_deployments, global_guardrails):
    pulumi.export(
        config.deployment_args.resource_name,
        deployment.id.apply(get_deployment_url),
    )
pulumi.export(
    settings_grader.deployment_args.resource_name,
    grading_deployment.id.apply(get_deployment_url),
)
pulumi.export(
    settings_rag.deployment_args.resource_name,
    rag_deployment.id.apply(get_deployment_url),
)
pulumi.export(
    settings_app_infra.app_resource_name,
    qa_application.application_url,
)
