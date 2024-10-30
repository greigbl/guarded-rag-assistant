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

# disable mypy:
# type: ignore

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pulumi_datarobot as datarobot
import pytest

try:
    from datarobot_drum.drum.drum import CMRunner
    from datarobot_drum.drum.runtime import DrumRuntime
    from datarobot_drum.runtime_parameters.runtime_parameters import (
        RuntimeParametersLoader,
    )
except ImportError:
    pass

sys.path.append("../")
from docsassist.schema import RAGInput
from infra.components.dr_credential import DRCredential


@pytest.fixture
def drum_installed():
    try:
        import datarobot_drum  # noqa: F401
    except ImportError:
        pytest.skip(
            "DIY RAG custom model tests requires datarobot_drum to be installed."
        )


class ExtendedDRCredential(DRCredential):
    def __init__(self, *args, **kwargs):
        # Store all arguments for later inspection
        self._init_kwargs = kwargs

        # Call the original __init__ method
        super().__init__(*args, **kwargs)

    def get_init_kwargs(self) -> Dict[str, Any]:
        """Return the arguments passed to __init__"""
        return self._init_kwargs


class ExtendedCustomModel(datarobot.CustomModel):
    def __init__(self, *args, **kwargs):
        # Store all arguments for later inspection
        self._init_kwargs = kwargs

        # Call the original __init__ method
        super().__init__(*args, **kwargs)

    def get_init_kwargs(self) -> Dict[str, Any]:
        """Return the arguments passed to __init__"""
        return self._init_kwargs


@pytest.fixture
def test_input(output_dir: Path) -> Path:
    with open(output_dir / "test_input.csv", "w") as f:
        f.write(
            """promptText,association_id,messages
"Tell me about DataRobot?","id42","[]"
"""
        )
    return str(output_dir / "test_input.csv")


@pytest.fixture
def output_dir() -> Path:
    path = Path("tests/output")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def test_input2(output_dir: Path) -> Path:
    rag_input = RAGInput(
        promptText="Which fruit did I mention just now?",
        association_id="id42",
        messages=[
            {"content": "Banana", "role": "user"},
            {
                "role": "assistant",
                "content": "Hi there! How can I assist you today?",
            },
        ],
    )
    data = rag_input.model_dump(mode="json", by_alias=True)
    data["messages"] = json.dumps(data["messages"])
    pd.DataFrame.from_records([data]).to_csv(
        output_dir / "test_input2.csv",
    )
    return str(output_dir / "test_input2.csv")


@pytest.fixture
def test_input3(output_dir: Path) -> Path:
    rag_input = RAGInput(
        promptText="Tell me about DataRobot?",
        association_id="id42",
        messages=[],
    )
    pd.DataFrame.from_records([rag_input.model_dump(by_alias=True)]).to_csv(
        output_dir / "test_input3.csv"
    )
    return str(output_dir / "test_input3.csv")


@pytest.fixture
def rag_custom_model(code_dir) -> ExtendedCustomModel:
    from infra import settings_rag
    from infra.settings_llm_credential import credential, credential_args

    llm_credential = ExtendedDRCredential(
        resource_name="llm-credential",
        credential=credential,
        credential_args=credential_args,
    )
    diy_rag_files = settings_rag.get_diy_rag_files(
        runtime_parameter_values=llm_credential.runtime_parameter_values,
    )

    rag_custom_model = ExtendedCustomModel(
        files=diy_rag_files,
        runtime_parameter_values=llm_credential.runtime_parameter_values,
        **settings_rag.custom_model_args.model_dump(mode="json", exclude_none=True),
    )
    return rag_custom_model


def run_drum(
    code_dir: str,
    input: str,
    output: str,
    #  runtime_params_file: str
):
    from infra.settings_llm_credential import credential

    with DrumRuntime() as runtime:
        options = Namespace(
            subparser_name="score",
            code_dir=code_dir,
            verbose=False,
            input=input,
            logging_level="info",
            docker=None,
            skip_deps_install=False,
            memory=None,
            output=output,
            show_perf=False,
            sparse_column_file=None,
            language=None,
            show_stacktrace=False,
            monitor=False,
            monitor_embedded=False,
            deployment_id=None,
            model_id=None,
            monitor_settings=None,
            allow_dr_api_access=False,
            webserver=None,
            api_token=None,
            gpu_predictor=None,
            triton_host="http://localhost",
            triton_http_port="8000",
            triton_grpc_port="8001",
            target_type="textgeneration",
            query=None,
            content_type=None,
            user_secrets_mount_path=None,
            user_secrets_prefix=None,
        )

        runtime.options = options
        if "runtime_params_file" in options and options.runtime_params_file:
            loader = RuntimeParametersLoader(
                options.runtime_params_file, options.code_dir
            )
            loader._yaml_content["OPENAI_API_KEY"]["apiToken"] = credential.api_key
            loader.setup_environment_variables()
        runtime.options = options
        runner = CMRunner(runtime)
        runner.run()


@pytest.fixture
def code_dir() -> str:
    code_dir = Path("deployment_diy_rag")
    code_dir.mkdir(exist_ok=True, parents=True)
    return code_dir


def test_diy_rag_custom_model(
    code_dir: str,
    test_input: Path,
    test_input2: Path,
    test_input3: Path,
    output_dir: Path,
    rag_mode: str,
    drum_installed,
) -> None:
    from infra.settings_rag import custom_model_args

    if rag_mode != "diy":
        pytest.skip("Skipping DIY RAG custom model tests")
    os.environ["TARGET_NAME"] = str(custom_model_args.target_name)
    # persisting output for investigation
    output = str(output_dir / "out1.csv")
    run_drum(
        code_dir=code_dir,
        input=test_input,
        output=output,
    )
    with open(output) as f:
        content = f.read()

    assert "DataRobot" in content

    output2 = str(output_dir / "out2.csv")
    run_drum(
        code_dir=code_dir,
        input=test_input2,
        output=output2,
    )
    with open(output2) as f:
        content2 = f.read()

    assert "banana" in content2.lower()

    output3 = str(output_dir / "out3.csv")
    run_drum(
        code_dir=code_dir,
        input=test_input3,
        output=output3,
    )
    with open(output3) as f:
        content3 = f.read()

    assert "DataRobot" in content3
