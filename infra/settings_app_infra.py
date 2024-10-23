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

from pathlib import Path

import datarobot as dr
import pulumi

from docsassist.i18n import LanguageCode, LocaleSettings
from infra.common.globals import GlobalRuntimeEnvironment
from infra.common.schema import ApplicationSourceArgs
from infra.settings_main import project_name


def ensure_app_settings(app_id: str) -> None:
    try:
        dr.client.get_client().patch(
            f"customApplications/{app_id}/",
            json={"allowAutoStopping": True},
            timeout=60,
        )
    except Exception:
        pulumi.warn("Could not enable autostopping for the Application")


_application_path = Path("frontend/")


source_files = [
    (str(f), str(f.relative_to(_application_path)))
    for f in _application_path.glob("**/*")
    if f.is_file()
]

source_files.extend(
    [
        ("docsassist/__init__.py", "docsassist/__init__.py"),
        ("docsassist/deployments.py", "docsassist/deployments.py"),
        ("docsassist/predict.py", "docsassist/predict.py"),
        ("docsassist/schema.py", "docsassist/schema.py"),
        ("docsassist/i18n.py", "docsassist/i18n.py"),
    ]
)

application_locale = LocaleSettings().app_locale

if application_locale != LanguageCode.EN:
    source_files.append(
        (
            f"docsassist/locale/{application_locale}/LC_MESSAGES/base.mo",
            f"docsassist/locale/{application_locale}/LC_MESSAGES/base.mo",
        )
    )

app_source_args = ApplicationSourceArgs(
    resource_name=f"Guarded RAG App Source [{project_name}]",
    base_environment_id=GlobalRuntimeEnvironment.PYTHON_39_STREAMLIT.value.id,
    files=source_files,
).model_dump(mode="json", exclude_none=True)

app_resource_name: str = f"Guarded RAG Application [{project_name}]"
