"""HTTP shape, parsing and error mapping for the OpenAI-compatible provider.

Every request is served by ``httpx.MockTransport``, so no socket is ever opened.
"""

from __future__ import annotations

import json

import httpx
import pytest

from dosubs import config
from dosubs.models import TranslationSettings
from dosubs.translate import OpenAICompatibleProvider, build_provider
from dosubs.translate.base import (
    TranslationAuthError,
    TranslationError,
    TranslationFormatError,
)

KEY = "sk-test-secret-0123456789"
CUES = [(1, "Hello."), (2, "How are you?")]
PAYLOAD = {"translations": [{"id": 1, "text": "你好。"}, {"id": 2, "text": "你好吗？"}]}
EXPECTED = {1: "你好。", 2: "你好吗？"}


def chat_response(content: str, *, usage: dict[str, int] | None = None) -> httpx.Response:
    body: dict[str, object] = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def recording_handler(
    content: str, *, seen: list[httpx.Request], status: int = 200, usage=None
):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200:
            return httpx.Response(
                status,
                json={"error": {"message": f"upstream said no ({status})", "type": "x"}},
            )
        return chat_response(content, usage=usage)

    return handler


def make_provider(
    handler,
    *,
    base_url: str = "https://api.deepseek.com",
    provider_name: str = "deepseek",
    api_key: str | None = KEY,
    **kwargs,
) -> OpenAICompatibleProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider(
        base_url=base_url,
        model="deepseek-flash",
        api_key=api_key,
        client=client,
        provider_name=provider_name,
        **kwargs,
    )


def body_of(request: httpx.Request) -> dict:
    return json.loads(request.content)


# --------------------------------------------------------------------------- #
# URL / payload shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.deepseek.com", "https://api.deepseek.com/chat/completions"),
        ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1/chat/completions"),
        ("https://api.openai.com/v1/", "https://api.openai.com/v1/chat/completions"),
        (
            "https://proxy.local/v1/chat/completions",
            "https://proxy.local/v1/chat/completions",
        ),
    ],
)
def test_endpoint_is_chat_completions(base_url: str, expected: str):
    seen: list[httpx.Request] = []
    provider = make_provider(recording_handler(json.dumps(PAYLOAD), seen=seen), base_url=base_url)

    assert provider.translate_window(CUES, source_language="en", target_language="zh-CN") == EXPECTED
    assert str(seen[0].url) == expected
    assert seen[0].method == "POST"


def test_request_body_and_auth_header():
    seen: list[httpx.Request] = []
    provider = make_provider(
        recording_handler(json.dumps(PAYLOAD), seen=seen),
        extra_instructions="Keep it informal.",
        temperature=0.4,
    )

    provider.translate_window(CUES, source_language="en", target_language="zh-CN")

    request = seen[0]
    body = body_of(request)
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert body["model"] == "deepseek-flash"
    assert body["temperature"] == 0.4
    assert body["stream"] is False
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}

    system, user = body["messages"]
    assert system["role"] == "system"
    assert "zh-CN" in system["content"]
    assert "en" in system["content"]
    assert "Keep it informal." in system["content"]
    assert "translations" in system["content"]
    assert user["role"] == "user"

    sent = json.loads(user["content"])
    assert sent == {
        "source_language": "en",
        "target_language": "zh-CN",
        "cues": [{"id": 1, "text": "Hello."}, {"id": 2, "text": "How are you?"}],
    }


def test_user_message_is_not_ascii_escaped():
    seen: list[httpx.Request] = []
    provider = make_provider(recording_handler(json.dumps(PAYLOAD), seen=seen))

    provider.translate_window([(1, "Ça va?")], source_language="fr", target_language="zh-CN")

    assert "Ça va?" in seen[0].content.decode("utf-8")


def test_thinking_is_sent_for_deepseek_only():
    seen: list[httpx.Request] = []
    provider = make_provider(
        recording_handler(json.dumps(PAYLOAD), seen=seen),
        base_url="http://127.0.0.1:11434/v1",
        provider_name="ollama",
        api_key=None,
    )

    provider.translate_window(CUES, source_language=None, target_language="zh-CN")

    body = body_of(seen[0])
    assert "thinking" not in body
    assert "authorization" not in seen[0].headers


def test_repr_never_shows_the_key():
    provider = make_provider(recording_handler(json.dumps(PAYLOAD), seen=[]))
    assert KEY not in repr(provider)
    assert KEY not in str(provider)


# --------------------------------------------------------------------------- #
# Reply parsing
# --------------------------------------------------------------------------- #


def test_fenced_json_reply_parses():
    content = "```json\n" + json.dumps(PAYLOAD, ensure_ascii=False) + "\n```"
    provider = make_provider(recording_handler(content, seen=[]))
    assert provider.translate_window(CUES, source_language="en", target_language="zh-CN") == EXPECTED


def test_prose_wrapped_reply_parses():
    content = "Sure, here is the JSON:\n" + json.dumps(PAYLOAD, ensure_ascii=False) + "\nEnjoy!"
    provider = make_provider(recording_handler(content, seen=[]))
    assert provider.translate_window(CUES, source_language="en", target_language="zh-CN") == EXPECTED


def test_mapping_style_reply_parses():
    content = json.dumps({"translations": {"1": "你好。", "2": "你好吗？"}}, ensure_ascii=False)
    provider = make_provider(recording_handler(content, seen=[]))
    assert provider.translate_window(CUES, source_language="en", target_language="zh-CN") == EXPECTED


def test_bare_mapping_reply_parses():
    content = json.dumps({"1": "你好。", "2": "你好吗？"}, ensure_ascii=False)
    provider = make_provider(recording_handler(content, seen=[]))
    assert provider.translate_window(CUES, source_language="en", target_language="zh-CN") == EXPECTED


def test_unparseable_reply_raises_format_error():
    provider = make_provider(recording_handler("I cannot translate that.", seen=[]))
    with pytest.raises(TranslationFormatError) as excinfo:
        provider.translate_window(CUES, source_language="en", target_language="zh-CN")
    assert "could not find a JSON object" in str(excinfo.value)


def test_reply_without_choices_raises_format_error():
    provider = make_provider(recording_handler(json.dumps({"error": "nope"}), seen=[]))
    with pytest.raises(TranslationFormatError):
        provider.translate_window(CUES, source_language="en", target_language="zh-CN")


def test_format_error_excerpt_is_short():
    provider = make_provider(recording_handler("x" * 5000, seen=[]))
    with pytest.raises(TranslationFormatError) as excinfo:
        provider.translate_window(CUES, source_language="en", target_language="zh-CN")
    assert len(str(excinfo.value)) < 400


def test_usage_is_recorded():
    usage = {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
    provider = make_provider(recording_handler(json.dumps(PAYLOAD), seen=[], usage=usage))
    provider.translate_window(CUES, source_language="en", target_language="zh-CN")
    assert provider.last_usage == usage


# --------------------------------------------------------------------------- #
# Transport failures
# --------------------------------------------------------------------------- #


def test_401_raises_auth_error():
    provider = make_provider(recording_handler("", seen=[], status=401))
    with pytest.raises(TranslationAuthError) as excinfo:
        provider.translate_window(CUES, source_language="en", target_language="zh-CN")
    assert "401" in str(excinfo.value)
    assert "upstream said no (401)" in str(excinfo.value)
    assert isinstance(excinfo.value, TranslationError)


def test_403_raises_auth_error():
    provider = make_provider(recording_handler("", seen=[], status=403))
    with pytest.raises(TranslationAuthError):
        provider.translate_window(CUES, source_language="en", target_language="zh-CN")


def test_500_raises_translation_error():
    provider = make_provider(recording_handler("", seen=[], status=500))
    with pytest.raises(TranslationError) as excinfo:
        provider.translate_window(CUES, source_language="en", target_language="zh-CN")
    assert not isinstance(excinfo.value, TranslationAuthError)
    assert "500" in str(excinfo.value)


def test_429_raises_translation_error_for_batch_to_retry():
    provider = make_provider(recording_handler("", seen=[], status=429))
    with pytest.raises(TranslationError) as excinfo:
        provider.translate_window(CUES, source_language="en", target_language="zh-CN")
    assert not isinstance(excinfo.value, TranslationAuthError)


def test_timeout_raises_translation_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = make_provider(handler, timeout_s=5.0)
    with pytest.raises(TranslationError) as excinfo:
        provider.translate_window(CUES, source_language="en", target_language="zh-CN")
    assert "timed out" in str(excinfo.value)


def test_connection_error_raises_translation_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    provider = make_provider(handler)
    with pytest.raises(TranslationError) as excinfo:
        provider.translate_window(CUES, source_language="en", target_language="zh-CN")
    assert "connection failed" in str(excinfo.value)


@pytest.mark.parametrize("status", [401, 429, 500])
def test_key_never_appears_in_errors(status: int):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"message": f"rejected Authorization: Bearer {KEY}"}}
        )

    provider = make_provider(handler)
    with pytest.raises(TranslationError) as excinfo:
        provider.translate_window(CUES, source_language="en", target_language="zh-CN")
    assert KEY not in str(excinfo.value)
    assert "***" in str(excinfo.value)


def test_key_echoed_in_reply_is_scrubbed():
    content = json.dumps({"translations": f"here is your key {KEY}"})
    provider = make_provider(recording_handler(content, seen=[]))
    with pytest.raises(TranslationFormatError) as excinfo:
        provider.translate_window(CUES, source_language="en", target_language="zh-CN")
    assert KEY not in str(excinfo.value)


# --------------------------------------------------------------------------- #
# check_connection
# --------------------------------------------------------------------------- #


def test_check_connection_reports_reachability():
    seen: list[httpx.Request] = []
    payload = {"translations": [{"id": 1, "text": "Hello there."}]}
    provider = make_provider(recording_handler(json.dumps(payload), seen=seen))

    assert provider.check_connection() == "deepseek/deepseek-flash reachable"
    sent = json.loads(body_of(seen[0])["messages"][1]["content"])
    assert sent["cues"] == [{"id": 1, "text": "Hello there."}]


def test_check_connection_raises_on_bad_key():
    provider = make_provider(recording_handler("", seen=[], status=401))
    with pytest.raises(TranslationAuthError):
        provider.check_connection()


# --------------------------------------------------------------------------- #
# build_provider
# --------------------------------------------------------------------------- #


def silent_client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))


def test_build_provider_unknown_preset(monkeypatch):
    monkeypatch.setattr(config, "get_api_key", lambda provider: None)
    with pytest.raises(TranslationError, match="unknown translation provider"):
        build_provider(TranslationSettings(provider="nope"), client=silent_client())


def test_build_provider_custom_needs_base_url_and_model(monkeypatch):
    monkeypatch.setattr(config, "get_api_key", lambda provider: None)
    settings = TranslationSettings(provider="custom", base_url="", model="")
    with pytest.raises(TranslationError, match="base_url"):
        build_provider(settings, client=silent_client())


def test_build_provider_custom_uses_settings(monkeypatch):
    monkeypatch.setattr(config, "get_api_key", lambda provider: None)
    settings = TranslationSettings(
        provider="custom", base_url="http://localhost:8000/v1", model="local-1"
    )
    provider = build_provider(settings, client=silent_client())
    assert provider.name == "custom"
    assert provider.model == "local-1"
    assert provider.endpoint == "http://localhost:8000/v1/chat/completions"


def test_build_provider_requires_key_for_deepseek(monkeypatch):
    monkeypatch.setattr(config, "get_api_key", lambda provider: None)
    with pytest.raises(TranslationAuthError, match="no API key"):
        build_provider(TranslationSettings(provider="deepseek"), client=silent_client())


def test_build_provider_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("DOSUBS_DEEPSEEK_API_KEY", "sk-from-env-1234567")
    seen: list[httpx.Request] = []
    client = httpx.Client(
        transport=httpx.MockTransport(recording_handler(json.dumps(PAYLOAD), seen=seen))
    )
    provider = build_provider(TranslationSettings(provider="deepseek"), client=client)

    assert provider.base_url == "https://api.deepseek.com"
    provider.translate_window(CUES, source_language="en", target_language="zh-CN")
    assert seen[0].headers["authorization"] == "Bearer sk-from-env-1234567"
    assert str(seen[0].url) == "https://api.deepseek.com/chat/completions"
    assert "sk-from-env-1234567" not in repr(provider)


def test_build_provider_ollama_needs_no_key_and_falls_back_to_preset(monkeypatch):
    monkeypatch.setattr(config, "get_api_key", lambda provider: None)
    settings = TranslationSettings(provider="ollama", base_url="", model="")
    provider = build_provider(settings, client=silent_client())
    assert provider.base_url == "http://127.0.0.1:11434/v1"
    assert provider.model == "qwen3:8b"
    assert "Authorization" not in provider._headers()


def test_build_provider_settings_override_preset():
    settings = TranslationSettings(
        provider="deepseek",
        base_url="https://proxy.local/v1",
        model="my-model",
        timeout_s=9.0,
    )
    provider = build_provider(
        settings, client=silent_client(), api_key="sk-explicit-key-1234"
    )
    assert provider.base_url == "https://proxy.local/v1"
    assert provider.model == "my-model"
    assert provider.timeout_s == 9.0


def test_local_endpoints_ignore_environment_proxies():
    """A system proxy must never intercept Ollama on localhost.

    Observed on a machine with a system proxy configured: the request went to the
    proxy and came back as ``HTTP 502`` instead of reaching the local server.
    """
    from dosubs.translate.openai_compat import _default_client, _is_local_endpoint

    assert _is_local_endpoint("http://127.0.0.1:11434/v1")
    assert _is_local_endpoint("http://localhost:11434/v1")
    assert _is_local_endpoint("http://[::1]:8080/v1")
    assert _is_local_endpoint("http://ollama.localhost/v1")
    assert not _is_local_endpoint("https://api.deepseek.com")
    assert not _is_local_endpoint("https://api.openai.com/v1")

    local = _default_client("http://127.0.0.1:11434/v1", 5.0)
    try:
        assert local.trust_env is False
        assert not local._mounts  # no proxy transports mounted
    finally:
        local.close()

    cloud = _default_client("https://api.deepseek.com", 5.0)
    try:
        assert cloud.trust_env is True
    finally:
        cloud.close()
