# Guarded RAG Assistant

The guarded RAG assistant is an easily customizable recipe for building a RAG-powered chatbot. 

In addition to creating a hosted, shareable user interface, the guarded RAG assistant provides:

* Business logic and LLM based guardrails.
* A predictive secondary model that evaluates response quality.
* GenAI-focused [custom metrics][custom-metrics].
* DataRobot MLOps hosting, monitoring, and governing the individual backend deployments.

> [!WARNING]
> Application Templates are intended to be starting points that provide guidance on how to develop, serve, and maintain AI applications.
> They require a developer or data scientist to adapt, and modify them to business requirements before being put into production.

![Using the Guarded RAG Assistant](https://s3.amazonaws.com/datarobot_public/drx/recipe_gifs/launch_gifs/guardedraghq-small.gif)

[custom-metrics]: https://docs.datarobot.com/en/docs/workbench/nxt-console/nxt-monitoring/nxt-custom-metrics.html

## Setup

1. If `pulumi` is not already installed, install the CLI following instructions [here](https://www.pulumi.com/docs/iac/download-install/). 
   After installing for the first time, restart your terminal and run:
   ```
   pulumi login --local  # omit --local to use Pulumi Cloud (requires separate account)
   ```

2. Clone the template repository.

   ```
   git clone https://github.com/datarobot-community/guarded-rag-assistant.git
   cd recipe-docsassist
   ```

3. Rename the file `.env.template` to `.env` in the root directory of the repo and populate your credentials.

   ```
   DATAROBOT_API_TOKEN=...
   DATAROBOT_ENDPOINT=...  # e.g. https://app.datarobot.com/api/v2
   OPENAI_API_KEY=...
   OPENAI_API_VERSION=...  # e.g. 2024-02-01
   OPENAI_API_BASE=...  # e.g. https://your_org.openai.azure.com/
   OPENAI_API_DEPLOYMENT_ID=...  # e.g. gpt-4
   PULUMI_CONFIG_PASSPHRASE=...  # required, choose an alphanumeric passphrase to be used for encrypting pulumi config
   ```
   
4. In a terminal run:
   ```
   python quickstart.py YOUR_PROJECT_NAME  # Windows users may have to use `py` instead of `python`
   ```

Advanced users desiring control over virtual environment creation, dependency installation, environment variable setup
and `pulumi` invocation see [here](#setup-for-advanced-users).


## Architecture Overview
![Guarded RAG Architecture](https://s3.amazonaws.com/datarobot_public/drx/recipe_gifs/rag_architecture.svg)

## Make changes

### Change the RAG documents

1. Replace `assets/datarobot_english_documentation_docsassist.zip` with a new zip file containing .pdf, .docx,
   .md, or .txt documents ([example alternative docs here](https://s3.amazonaws.com/datarobot_public_datasets/ai_accelerators/acme_corp_company_policies_source_business_victoria_templates.zip)).
3. Update the `rag_documents` setting in `infra/settings_main.py` to specify the local path to the
   new zip file.
4. Run `pulumi up` to update your stack.

### Change the LLM

1. Modify your `.env`.
   ```
   GOOGLE_SERVICE_ACCOUNT=''  # insert json service key between the single quotes, newlines are OK
   GOOGLE_REGION=...  # default is 'us-west1'
   ```
2. Update your environment and install `google-auth`.
   ```
   source set_env.sh  # On windows use `set_env.bat`
   pip install google-auth
   ```
3. Update the credential type to be provisioned in `infra/settings_llm_credential.py`.
   ```
   # credential = AzureOpenAICredentials()
   # credential.test()
   from docsassist.credentials import GoogleLLMCredentials
   credential = GoogleLLMCredentials()
   credential.test('gemini-1.5-flash-001')  # select a model for validating the credential
   ```
4. Configure a Gemini blueprint to be provisioned in `infra/settings_rag.py`.
   ```
   # llm_id=GlobalLLM.AZURE_OPENAI_GPT_3_5_TURBO,
   llm_id=GlobalLLM.GOOGLE_GEMINI_1_5_FLASH,
   ```
5. Run `pulumi up` to update your stack.
   
### Fully custom front-end
1. Edit `infra/settings_main.py` and update `application_type` to `ApplicationType.DIY`
   - Optionally, update `APP_LOCALE` in `docsassist/i18n.py` to toggle the language.
     Supported locales include French (fr_FR), Spanish (es_LA), Korean (ko_KR), and
     Brazilian Portuguese (pt_BR) in addition to the English default (en_US).
2. Run `pulumi up` to update your stack with the example custom Streamlit frontend,
3. After provisioning the stack at least once, you can also edit and test the Streamlit
   front-end locally using `streamlit run app.py` from the `frontend/` directory (don't
   forget to initialize your environment using `set_env`).

### Fully custom RAG chunking, vectorization and retrieval
1. Install additional requirements (e.g. FAISS, HuggingFace).
   ```
   source set_env.sh  # On windows use `set_env.bat`
   pip install -r requirements-extra.txt
   ```
2. Edit `infra/settings_main.py` and update `rag_type` to `RAGType.DIY`.
3. Run `pulumi up` to update your stack with the example custom RAG logic.
4. Edit `data_science/build_rag.ipynb` to customize the doc chunking, vectorization logic.
5. Edit `deployment_diy_rag/custom.py` to customize the retrieval logic & LLM call.
6. Run `pulumi up` to update your stack.

## Share results

1. Log into the DataRobot application.
2. Navigate to **Registry > Applications**.
3. Navigate to the application you want to share, open the actions menu, and select **Share** from the dropdown.

## Delete all provisioned resources
```
pulumi down
```

## Setup for advanced users
For manual control over the setup process adapt the following steps for MacOS/Linux to your environent:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source set_env.sh
pulumi stack init YOUR_PROJECT_NAME
pulumi up 
```
e.g. for Windows/conda/cmd.exe this would be:
```
conda create --prefix .venv pip
conda activate .\.venv
pip install -r requirements.txt
set_env.bat
pulumi stack init YOUR_PROJECT_NAME
pulumi up 
```
For projects that will be maintained, DataRobot recommends forking the repo so upstream fixes and improvements can be merged in the future.