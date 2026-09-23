import json

import pytest

from browser_agent.browser import validate_url
from browser_agent.state import RunLog, without_images


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///secret",
        "data:text/html,hi",
        "https://user:pass@example.com",
        "/invented",
    ],
)
def test_rejects_non_web_and_credential_urls(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_checkpoint_redacts_api_key_and_excludes_images(tmp_path):
    key = "private-test-api-key"
    log = RunLog(tmp_path, key)
    log.checkpoint(
        key,
        "",
        [[{"role": "user", "content": [{"type": "input_image", "image_url": "data:large-image"}]}]],
        [],
        "running",
    )
    text = (tmp_path / "checkpoint.json").read_text(encoding="utf-8")
    assert key not in text and "data:large-image" not in text
    assert json.loads(text)["task"] == "[REDACTED]"


def test_image_pruning_does_not_mutate_live_messages():
    items = [{"role": "user", "content": [{"type": "input_image", "image_url": "data:test"}]}]
    assert without_images(items)[0]["content"][0]["type"] == "input_text"
    assert items[0]["content"][0]["type"] == "input_image"
