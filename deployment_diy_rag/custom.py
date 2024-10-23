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
import json
import os
import sys
import traceback

import pandas as pd
import yaml
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import (
    create_history_aware_retriever,
)
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.callbacks import get_openai_callback
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import AzureChatOpenAI
from pandas import DataFrame

sys.path.append("../")
from docsassist.credentials import AzureOpenAICredentials
from docsassist.schema import PROMPT_COLUMN_NAME, TARGET_COLUMN_NAME, RAGModelSettings


def get_chain(
    input_dir, credentials: AzureOpenAICredentials, model_settings: RAGModelSettings
):
    """Instantiate the RAG chain."""
    embedding_function = SentenceTransformerEmbeddings(
        model_name=model_settings.embedding_model_name,
        cache_folder=input_dir + "/sentencetransformers",
    )
    db = FAISS.load_local(
        folder_path=input_dir + "/faiss_db",
        embeddings=embedding_function,
        allow_dangerous_deserialization=True,
    )

    llm = AzureChatOpenAI(
        deployment_name=credentials.azure_deployment,
        azure_endpoint=credentials.azure_endpoint,
        openai_api_version=credentials.api_version,
        openai_api_key=credentials.api_key,
        model_name=credentials.azure_deployment,
        temperature=model_settings.temperature,
        verbose=True,
        max_retries=model_settings.max_retries,
        request_timeout=model_settings.request_timeout,
    )
    retriever = VectorStoreRetriever(
        vectorstore=db,
    )
    system_template = model_settings.stuff_prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question
    qa_system_prompt = system_template
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # Below we use create_stuff_documents_chain to feed all retrieved context
    # into the LLM. Note that we can also use StuffDocumentsChain and other
    # instances of BaseCombineDocumentsChain.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def load_model(input_dir):
    """Load vector database and prepare chain."""
    with open(os.path.join(input_dir, RAGModelSettings.filename())) as f:
        model_settings = RAGModelSettings.model_validate(yaml.safe_load(f))
    credentials = AzureOpenAICredentials()
    chain = get_chain(input_dir, credentials=credentials, model_settings=model_settings)
    return chain, model_settings


def score(data: pd.DataFrame, model: tuple[Runnable, RAGModelSettings], **kwargs):
    """ "Orchestrate a RAG completion with our vector database."""

    chain, model_settings = model

    full_result_dict: dict[str, list] = {TARGET_COLUMN_NAME: []}

    for i, row in data.iterrows():
        question = row[PROMPT_COLUMN_NAME]
        chat_history = []
        if "messages" in row:
            messages = row["messages"]
            messages = json.loads(messages)
            for _, a in enumerate(messages):
                message_dict = a
                if message_dict["role"] == "user":
                    message = HumanMessage.validate(message_dict)
                else:
                    message = AIMessage.validate(message_dict)
                chat_history.append(message)

        try:
            with get_openai_callback():
                chain_output = chain.invoke(
                    {
                        "input": question,
                        "chat_history": chat_history,
                    }
                )
            full_result_dict[TARGET_COLUMN_NAME].append(chain_output["answer"])
            for i, doc in enumerate(chain_output["context"]):
                if f"CITATION_CONTENT_{i}" not in full_result_dict:
                    full_result_dict[f"CITATION_CONTENT_{i}"] = []
                if f"CITATION_SOURCE_{i}" not in full_result_dict:
                    full_result_dict[f"CITATION_SOURCE_{i}"] = []
                if f"CITATION_PAGE_{i}" not in full_result_dict:
                    full_result_dict[f"CITATION_PAGE_{i}"] = []
                full_result_dict[f"CITATION_CONTENT_{i}"].append(doc.page_content)
                full_result_dict[f"CITATION_SOURCE_{i}"].append(
                    doc.metadata.get("source", "")
                )
                full_result_dict[f"CITATION_PAGE_{i}"].append(
                    doc.metadata.get("page", "")
                )

        except Exception:
            full_result_dict[TARGET_COLUMN_NAME].append(traceback.format_exc())

    return DataFrame(full_result_dict)
