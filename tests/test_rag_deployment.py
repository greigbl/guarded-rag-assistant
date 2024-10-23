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

import logging

import datarobot as dr

from docsassist.schema import PROMPT_COLUMN_NAME

logger = logging.getLogger(__name__)


def test_rag_deployment_prediction(
    pulumi_up, make_prediction, rag_deployment_id, association_id
):
    def extract_references(rag_response_dict):
        if "references" in rag_response_dict:
            return rag_response_dict["references"]

        references = []
        for column in sorted(rag_response_dict.keys()):
            if "CITATION_CONTENT" in column:
                doc = {"page_content": rag_response_dict[column], "metadata": {}}
                source = rag_response_dict[column.replace("CONTENT", "SOURCE")]
                if column.replace("CONTENT", "PAGE") in rag_response_dict:
                    page = rag_response_dict[column.replace("CONTENT", "PAGE")]
                    doc["metadata"]["source"] = f"{source}, page {page}"
                else:
                    doc["metadata"]["source"] = source
                references.append(doc)
        return references

    prompt_feature_name = PROMPT_COLUMN_NAME
    msg = f"RAG Deployment ID: {rag_deployment_id}"
    logger.info(msg)

    rag_input = [
        {
            prompt_feature_name: "tell me about DataRobot",
            "association_id": association_id,
        }
    ]
    rag_response_dict = make_prediction(rag_input, rag_deployment_id)

    rag_deployment = dr.Deployment.get(rag_deployment_id)
    rag_response = {
        "answer": rag_response_dict[
            f"{rag_deployment.model['target_name']}_PREDICTION"
        ],
        "references": extract_references(rag_response_dict),
    }
    if "usage" in rag_response_dict:
        rag_response["usage"] = rag_response_dict["usage"]

    assert len(rag_response["answer"]) > 0 and len(rag_response["references"]) > 0
