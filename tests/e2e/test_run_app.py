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

import time

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By

from tests.e2e.utils import wait_for_element_to_be_visible


@pytest.mark.usefixtures("check_if_logged_in")
def test_app_loaded(browser: webdriver.Chrome, get_app_url: str) -> None:
    browser.get(get_app_url)

    assert wait_for_element_to_be_visible(browser, By.ID, "root")
    assert wait_for_element_to_be_visible(browser, By.ID, "q-a-chat-application")
    assert (
        "Q&A Chat Application"
        in browser.find_element(By.ID, "q-a-chat-application").text
    )


@pytest.mark.usefixtures("check_if_logged_in")
def test_input_message(browser: webdriver.Chrome, get_app_url: str) -> None:
    # browser.get(get_app_url)
    # time.sleep(2)
    # textarea with placeholder "Your message"
    # button disabled
    assert wait_for_element_to_be_visible(browser, By.CSS_SELECTOR, "textarea")
    send_button = browser.find_element(
        By.CSS_SELECTOR, '[data-testid="stChatInputSubmitButton"]'
    )
    assert send_button.get_attribute("disabled")
    input_message = browser.find_element(By.CSS_SELECTOR, "textarea")
    assert input_message.get_attribute("placeholder") == "Send a prompt"
    input_message.send_keys("Hello, world!")
    assert input_message.get_attribute("value") == "Hello, world!"
    # button not disabled
    assert not send_button.get_attribute("disabled")


@pytest.mark.usefixtures("check_if_logged_in")
def test_send_message(browser: webdriver.Chrome, get_app_url: str) -> None:
    # get button by data-testid attribute
    send_button = browser.find_element(
        By.CSS_SELECTOR, '[data-testid="stChatInputSubmitButton"]'
    )
    assert not send_button.get_attribute("disabled")
    input_message = browser.find_element(By.CSS_SELECTOR, "textarea")
    input_message.send_keys("Hello, world!")
    send_button.click()
    # wait for message to be sent and resopnse to be received
    time.sleep(4)
    # get count of messages
    messages = browser.find_elements(By.CSS_SELECTOR, ".stChatMessage")
    assert len(messages) == 2
