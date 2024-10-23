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

import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Tuple

import datarobot as dr
import pandas as pd
from datarobot.models.deployment.deployment import Deployment
from datarobot_predict.deployment import PredictionResult, predict
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import ValidationError

from docsassist.deployments import GradingDeployment, RAGDeployment  # noqa: E402
from docsassist.schema import (  # noqa: E402
    Grade,
    GraderOutput,
    RAGInput,
    RAGOutput,
)

try:
    rag_deployment_id = RAGDeployment().id
    grading_deployment_id = GradingDeployment().id
except ValidationError as e:
    raise ValueError(
        (
            "Unable to load DataRobot deployment ids. If running locally, verify you have selected "
            "the correct stack and that it is active using `pulumi stack output`. "
            "If running in DataRobot, verify your runtime parameters have been set correctly."
        )
    ) from e


@dataclass
class DeploymentInfo:
    deployment: Deployment
    target_name: str


def _get_deployment_info(deployment_id: str) -> DeploymentInfo:
    deployment = dr.Deployment.get(deployment_id)  # type: ignore[attr-defined]
    target_name = deployment.model["target_name"]  # type: ignore[index]
    return DeploymentInfo(deployment, str(target_name))


def _predict_with_retry(
    deployment: Deployment,
    data_frame: pd.DataFrame,
    max_wait_seconds: int = 300,
    retry_interval_seconds: int = 5,
) -> PredictionResult:
    start_time = time.time()
    while True:
        try:
            prediction = predict(deployment, data_frame=data_frame)
            return prediction
        except dr.errors.ServerError as e:
            if "Inference server is starting" in str(e):
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_seconds:
                    raise TimeoutError(
                        f"Server did not start within {max_wait_seconds} seconds"
                    )
                print(
                    f"Server is starting. Retrying in {retry_interval_seconds} seconds..."
                )
                time.sleep(retry_interval_seconds)
            else:
                # If it's a different ServerError, re-raise it
                raise


# TODO: validate interface schemas are cleanly serializable
def predict_grade(data: RAGOutput, association_id: str) -> GraderOutput:
    grading_deployment_info = _get_deployment_info(grading_deployment_id)
    grading_deployment = grading_deployment_info.deployment
    target_name = grading_deployment_info.target_name

    df = data.to_dataframe()

    df.rename(columns={"question": "prompt", "answer": "response"}, inplace=True)

    df["association_id"] = association_id
    response_df = _predict_with_retry(grading_deployment, data_frame=df).dataframe

    response_dict = response_df.to_dict(orient="records")[0]
    response_dict["__target"] = target_name
    response = GraderOutput.model_validate(response_dict)

    return response


def submit_grade(grade: Grade, association_id: str) -> None:
    data = [{"association_id": association_id, "actual_value": grade}]

    deployment = dr.Deployment.get(grading_deployment_id)  # type: ignore[attr-defined]
    deployment.submit_actuals(data)


def get_rag_completion(
    question: str, messages: Iterable[ChatCompletionMessageParam]
) -> Tuple[RAGOutput, str]:
    """Retrieve predictions from a DataRobot RAG deployment and DataRobot guard deployment"""
    rag_deployment_info = _get_deployment_info(rag_deployment_id)
    rag_deployment = rag_deployment_info.deployment
    target_name = rag_deployment_info.target_name

    association_id = f"{uuid.uuid4().hex}_{datetime.now()}"
    data = RAGInput(
        promptText=question,
        messages=messages,
        association_id=association_id,
    ).model_dump(mode="json", by_alias=True)
    data["messages"] = json.dumps(data["messages"])
    rag_input = pd.DataFrame.from_records([data])
    logging.info(
        f"Rag Input: {json.dumps(rag_input.to_dict(orient='records'), indent=4)}"
    )

    rag_response_df = _predict_with_retry(
        rag_deployment, data_frame=rag_input
    ).dataframe
    rag_response_df.columns = rag_response_df.columns.str.replace(
        "_(PREDICTION|OUTPUT)$", "", regex=True
    )
    rag_response_dict = rag_response_df.to_dict(orient="records")[0]
    rag_response_dict["__target"] = target_name

    rag_output = RAGOutput.model_validate(rag_response_dict)

    return rag_output, association_id
