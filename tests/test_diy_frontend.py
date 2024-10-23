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

# mypy: ignore-errors

import contextlib
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, get_args

import datarobot as dr
import pytest
from streamlit.testing.v1 import AppTest

from docsassist.schema import PROMPT_COLUMN_NAME, Grade

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def working_rag_deployment(
    pulumi_up, make_prediction, rag_deployment_id, association_id
):
    prompt_feature_name = PROMPT_COLUMN_NAME

    rag_input = [
        {
            prompt_feature_name: "tell me about DataRobot",
            "association_id": association_id,
        }
    ]
    rag_response_dict = make_prediction(rag_input, rag_deployment_id)

    rag_deployment = dr.Deployment.get(rag_deployment_id)
    rag_response = {
        "answer": rag_response_dict[f"{rag_deployment.model['target_name']}_PREDICTION"]
    }

    assert len(rag_response["answer"]) > 0


@contextlib.contextmanager
def cd(new_dir: Path) -> Any:
    """Changes the current working directory to the given path and restores the old directory on exit."""
    prev_dir = os.getcwd()

    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)


@pytest.fixture
def grades() -> tuple[Any, ...]:
    return get_args(Grade)


@pytest.fixture
def application(
    app_mode: str,
    pulumi_up: Any,
    subprocess_runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
    working_rag_deployment,
) -> Any:
    import docsassist.predict

    stack_name = subprocess.check_output(
        ["pulumi", "stack", "--show-name"],
        text=True,
    ).split("\n")[0]

    if app_mode == "dr":
        pytest.skip("Skipping DR frontend tests")

    with cd("frontend"):  # type: ignore[arg-type]
        # we need to select the stack again due to changing the directory
        # this loses the stack information in DR codespaces
        subprocess_runner(
            ["pulumi", "stack", "select", stack_name, "--non-interactive"]
        )

        # and ensure we can access `frontend` as if we were running from inside
        sys.path.append(".")
        logger.info(subprocess.check_output(["pulumi", "stack", "output"]))

        # reloading to ensure we don't use the deployment ID's of previous runs
        importlib.reload(docsassist.predict)
        yield AppTest.from_file("app.py", default_timeout=30)


@pytest.fixture
def app_prompt() -> str:
    return "What is the DataRobot?"


@pytest.fixture
def app_post_prompt(application: AppTest, app_prompt: str) -> AppTest:
    at = application.run()
    at.chat_input[0].set_value(app_prompt).run(timeout=60)
    return at


def test_produces_a_response(app_post_prompt: AppTest, app_prompt: str) -> None:
    response = app_post_prompt.session_state.response
    conversation_history_markdown = [
        i.value
        for i in app_post_prompt.markdown
        if '<div class="message-content">' in i.value
    ]
    assert "platform" in response.completion.lower()
    assert "DataRobot" in conversation_history_markdown[0]
    assert "platform" in conversation_history_markdown[1].lower()
    assert len(conversation_history_markdown) == 2


def test_reports_citations(app_post_prompt: AppTest) -> None:
    references = app_post_prompt.session_state.response.references
    citation_markdown = [i.value for i in app_post_prompt.expander[0].markdown]
    for i in range(len(references)):
        assert f"**Reference {i+1}:**" in citation_markdown

    assert len(references) > 0


def test_reports_grade(app_post_prompt: AppTest, grades: tuple[Any, ...]) -> None:
    grade = app_post_prompt.header[0].value
    assert grade.replace("**Response Grade**: ", "") in grades


def test_submit_feedback(app_post_prompt: AppTest, grades: tuple[Any, ...]) -> None:
    at = app_post_prompt
    at.button(f"button_{grades[0]}").click().run(timeout=30)

    assert len(at.button) == len(grades)
    assert at.success[0].value == "Thank you for your rating!"
