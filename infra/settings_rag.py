# Copyright 2024 DataRobot, Inc. and its affiliates.
# All rights reserved.
# DataRobot, Inc.
# This is proprietary source code of DataRobot, Inc. and its
# affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from __future__ import annotations

import pathlib
import textwrap

import datarobot as dr
import pulumi_datarobot as datarobot
from jinja2 import BaseLoader, Environment
from pydantic import BaseModel

from docsassist.schema import RAGModelSettings
from .common.globals import GlobalLLM
from .common.schema import (
    ChunkingParameters,
    CustomModelArgs,
    DatasetArgs,
    DeploymentArgs,
    LLMBlueprintArgs,
    LLMSettings,
    PlaygroundArgs,
    RegisteredModelArgs,
    VectorDatabaseArgs,
    VectorDatabaseSettings,
)
from .settings_main import (
    RAGType,
    core,
    default_prediction_server_id,
    project_name,
    runtime_environment_moderations,
)

custom_model_args = CustomModelArgs(
    resource_name="rag-custom-model",
    name="RAG Custom Model",
    base_environment_id=runtime_environment_moderations.id,
    target_name="completion",
    target_type=dr.enums.TARGET_TYPE.TEXT_GENERATION,
)

registered_model_args = RegisteredModelArgs(
    resource_name="rag-registered-model",
    name=f"RAG Registered Model [{project_name}]",
)


deployment_args = DeploymentArgs(
    resource_name="rag-deployment",
    label="RAG Deployment",
    association_id_settings=datarobot.DeploymentAssociationIdSettingsArgs(
        column_names=["association_id"],
        auto_generate_id=False,
        required_in_prediction_requests=True,
    ),
    predictions_settings=None
    if default_prediction_server_id
    else datarobot.DeploymentPredictionsSettingsArgs(
        min_computes=0, max_computes=1, real_time=True
    ),
    predictions_data_collection_settings=datarobot.DeploymentPredictionsDataCollectionSettingsArgs(
        enabled=True,
    ),
)

if core.rag_type == RAGType.DR:
    # if providing DIY RAG logic, these settings are N/A
    playground_args = PlaygroundArgs(
        resource_name="rag-playground", name="RAG Playground"
    )

    dataset_args = DatasetArgs(
        resource_name="rag-dataset",
        name="RAG Dataset",
        file_path=core.rag_documents,
    )
    vector_database_args = VectorDatabaseArgs(
        resource_name="rag-vector-database",
        name="RAG Vector Database",
        chunking_parameters=ChunkingParameters(
            embedding_model=dr.enums.VectorDatabaseEmbeddingModel.JINA_EMBEDDING_T_EN_V1,
            chunk_size=512,
            chunk_overlap_percentage=20,
        ),
    )

    llm_blueprint_args = LLMBlueprintArgs(
        resource_name="rag-llm-blueprint",
        name="RAG LLM Blueprint",
        llm_id=GlobalLLM.AZURE_OPENAI_GPT_3_5_TURBO,
        llm_settings=LLMSettings(
            max_completion_length=512,
            system_prompt=textwrap.dedent(
                """\
                Use the following pieces of context to answer the user's question.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                ----------------
                {context}"""
            ),
        ),
        vector_database_settings=VectorDatabaseSettings(
            max_documents_retrieved_per_prompt=10,
            max_tokens=512,
        ),
    )

elif core.rag_type == RAGType.DIY:

    class DIYRAGNotebookOutput(BaseModel):
        """Schema for DIY RAG notebook output paths."""

        vdb: pathlib.Path
        embedding_model: pathlib.Path
        rag_settings: pathlib.Path

    diy_rag_deployment_path = pathlib.Path("deployment_diy_rag/")
    diy_rag_nb = pathlib.Path("notebooks/build_rag.ipynb")
    diy_rag_nb_output = DIYRAGNotebookOutput(
        vdb=diy_rag_deployment_path / "faiss_db",
        embedding_model=diy_rag_deployment_path / "sentencetransformers",
        rag_settings=diy_rag_deployment_path / RAGModelSettings.filename(),
    )

    def get_diy_rag_files(
        runtime_parameter_values: list[datarobot.CustomModelRuntimeParameterValueArgs],
    ) -> list[tuple[str, str]]:
        """Prepare list of files to be uploaded to DIY RAG deployment."""
        llm_runtime_parameter_specs = "\n".join(
            [
                textwrap.dedent(f"""\
                - fieldName: {param.key}
                  type: {param.type}""")
                for param in runtime_parameter_values
            ]
        )

        with open(diy_rag_deployment_path / "model-metadata.yaml.jinja") as f:
            template = Environment(loader=BaseLoader()).from_string(f.read())
        with open(diy_rag_deployment_path / "model-metadata.yaml", "w") as f:
            runtime_parameters = template.render(
                custom_model_name=custom_model_args.name,
                target_type=custom_model_args.target_type,
                runtime_parameters=llm_runtime_parameter_specs,
            )
            f.write(runtime_parameters)

        diy_files = [
            (str(f), str(f.relative_to(diy_rag_deployment_path)))
            for f in diy_rag_deployment_path.glob("**/*")
            if f.is_file() and f.name not in ("README.md", "model-metadata.yaml.jinja")
        ] + [
            (
                "docsassist/__init__.py",
                "docsassist/__init__.py",
            ),
            ("docsassist/schema.py", "docsassist/schema.py"),
            ("docsassist/credentials.py", "docsassist/credentials.py"),
        ]
        return diy_files
