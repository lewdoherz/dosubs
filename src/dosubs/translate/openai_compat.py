"""Translation through any OpenAI-compatible ``/chat/completions`` endpoint.

Uses raw httpx instead of the ``openai`` SDK: we need one POST with one shape,
and the SDK would add a dependency (plus its own retry/streaming machinery) for
nothing. The reply is asked for as ``json_object`` because every provider we
target supports it and it removes a class of parse failures.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from typing import Any

import httpx

from .. import config, logging_setup
from ..models import PROVIDER_PRESETS, TranslationSettings
from .base import TranslationAuthError, TranslationError, TranslationFormatError

__all__ = ["OpenAICompatibleProvider", "build_provider"]

log = logging.getLogger(__name__)

#: Enough of a bad reply to recognise it in the log without dumping a whole prompt.
_EXCERPT_CHARS = 200

#: Providers that need ``thinking`` explicitly disabled. DeepSeek-V4.1 turns
#: thinking ON by default, and while it is on ``temperature`` is ignored.
_THINKING_PROVIDERS = frozenset({"deepseek"})


def _excerpt(value: object) -> str:
    text = value if isinstance(value, str) else str(value)
    return text[:_EXCERPT_CHARS]


def _system_prompt(
    target_language: str, source_language: str | None, extra_instructions: str
) -> str:
    lines = [
        f"You translate subtitle cues for dubbing and subtitles into {target_language}.",
        (
            f"Write {target_language} the way a native speaker would say it out loud: "
            "natural, conversational and matching the speaker's register."
        ),
        (
            "Preserve names, numbers, units and recurring terminology exactly, and "
            "translate a given term the same way everywhere in the window."
        ),
        (
            "Never add information that is not in the source, and never explain, "
            "annotate or transliterate instead of translating."
        ),
        (
            "Keep every cue short and self-contained so it can stand alone as one "
            "subtitle line."
        ),
    ]
    if source_language:
        lines.append(f"The source language is {source_language}.")
    if extra_instructions.strip():
        lines.append(extra_instructions.strip())
    lines.append(
        'Reply with JSON only: {"translations": [{"id": <int>, "text": "<translation>"}]} '
        "- exactly one entry per input id, no timestamps, no commentary."
    )
    return "\n".join(lines)


def _first_json_object(text: str) -> Any:
    """Decode the first JSON object/array in ``text``, tolerating fences and prose."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        return value
    raise ValueError("no JSON object in reply")


def _as_id(value: object) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        pass
    try:  # some endpoints echo ids as JSON numbers like 1.0
        return int(float(str(value).strip()))
    except (TypeError, ValueError) as exc:
        raise TranslationFormatError(f"translation entry has a non-numeric id: {value!r}") from exc


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, list):  # content-parts style reply
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in value
            if isinstance(part, (str, dict))
        )
    if isinstance(value, (int, float, bool)):
        return str(value)
    raise TranslationFormatError(f"translation entry has a non-text value: {_excerpt(value)}")


def _parse_translations(reply: str) -> dict[int, str]:
    """Map a raw model reply onto cue ids, accepting the shapes models emit."""
    try:
        payload = _first_json_object(reply)
    except ValueError as exc:
        raise TranslationFormatError(
            f"could not find a JSON object in the reply: {_excerpt(reply)}"
        ) from exc
    if not isinstance(payload, dict):
        raise TranslationFormatError(f"reply is not a JSON object: {_excerpt(reply)}")

    entries = payload.get("translations")
    if entries is None:
        # Some local models drop the wrapper and return {"1": "...", ...} directly.
        if payload and all(str(key).strip().lstrip("-").isdigit() for key in payload):
            entries = payload
        else:
            raise TranslationFormatError(
                f"reply has no 'translations' field: {_excerpt(reply)}"
            )

    translations: dict[int, str] = {}
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                raise TranslationFormatError(
                    f"translation entry is not an object: {_excerpt(entry)}"
                )
            translations[_as_id(entry.get("id"))] = _as_text(entry.get("text"))
    elif isinstance(entries, dict):
        for key, value in entries.items():
            if isinstance(value, dict):
                value = value.get("text")
            translations[_as_id(key)] = _as_text(value)
    else:
        raise TranslationFormatError(
            f"'translations' is neither a list nor an object: {_excerpt(entries)}"
        )

    if not translations:
        raise TranslationFormatError(f"reply contained no translations: {_excerpt(reply)}")
    return translations


def _message_content(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TranslationFormatError(
            f"reply has no choices[0].message.content: {_excerpt(payload)}"
        ) from exc
    content = _as_text(content)
    if not content.strip():
        raise TranslationFormatError(f"reply content was empty: {_excerpt(payload)}")
    return content


def _error_detail(response: httpx.Response) -> str:
    """``': message'`` from a JSON error body, or an excerpt of whatever came back."""
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return f": {_excerpt(text)}" if text else ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return f": {error['message']}"
        if isinstance(error, str) and error:
            return f": {error}"
        for key in ("message", "detail"):
            if isinstance(payload.get(key), str) and payload[key]:
                return f": {payload[key]}"
    return f": {_excerpt(payload)}"


_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0", "[::1]"})


def _is_local_endpoint(base_url: str) -> bool:
    host = urllib.parse.urlsplit(base_url).hostname or ""
    return host.lower() in _LOCAL_HOSTS or host.lower().endswith(".localhost")


def _default_client(base_url: str, timeout_s: float) -> httpx.Client:
    """Build the HTTP client, keeping environment proxies away from local endpoints.

    ``httpx`` trusts ``HTTP_PROXY``/``HTTPS_PROXY`` by default, so on a machine with a
    system proxy configured (a very common Windows setup) requests to a local Ollama
    server would be sent to the proxy instead of to localhost — observed here as
    ``HTTP 502`` from the proxy. Local endpoints are therefore contacted directly,
    while cloud providers keep honouring the environment.
    """
    if _is_local_endpoint(base_url):
        return httpx.Client(timeout=timeout_s, follow_redirects=True, trust_env=False)
    return httpx.Client(timeout=timeout_s, follow_redirects=True)


class OpenAICompatibleProvider:
    """A chat-completions translator for one model on one endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.2,
        timeout_s: float = 120.0,
        extra_instructions: str = "",
        client: httpx.Client | None = None,
        provider_name: str = "openai-compatible",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider_name = provider_name
        self.name = provider_name
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.extra_instructions = extra_instructions
        self.last_usage: dict[str, int] = {}
        self._api_key = api_key or None
        self._owns_client = client is None
        self._client = client or _default_client(base_url, timeout_s)
        if self._api_key:
            # Belt and braces: the log filter scrubs this if it ever reaches a record.
            logging_setup.register_secret(self._api_key)

    # -- plumbing ---------------------------------------------------------- #

    @property
    def label(self) -> str:
        return f"{self.provider_name}/{self.model}"

    @property
    def endpoint(self) -> str:
        """``base``, ``base/v1`` and ``base/chat/completions`` all work."""
        base = self.base_url.rstrip("/")
        if not base.endswith("/chat/completions"):
            base = f"{base}/chat/completions"
        return base

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatibleProvider(base_url={self.base_url!r}, model={self.model!r})"

    def _scrub(self, text: str) -> str:
        """Replace our own key wherever it shows up in server-supplied text."""
        if self._api_key:
            return text.replace(self._api_key, "***")
        return text

    def _scrubbed(self, exc: TranslationError) -> TranslationError:
        """Re-wrap ``exc`` if the endpoint echoed our key back at us."""
        message = self._scrub(str(exc))
        return exc if message == str(exc) else type(exc)(message)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _body(
        self, items: list[tuple[int, str]], source_language: str | None, target_language: str
    ) -> dict[str, Any]:
        user = {
            "source_language": source_language,
            "target_language": target_language,
            "cues": [{"id": cue_id, "text": text} for cue_id, text in items],
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": _system_prompt(
                        target_language, source_language, self.extra_instructions
                    ),
                },
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if self.provider_name in _THINKING_PROVIDERS:
            body["thinking"] = {"type": "disabled"}
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(self.endpoint, json=body, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise TranslationError(
                f"{self.label}: request timed out after {self.timeout_s:g}s"
            ) from exc
        except httpx.HTTPError as exc:
            # httpx errors carry the URL, never the request headers, so no key leaks.
            raise TranslationError(
                f"{self.label}: connection failed ({type(exc).__name__}): {exc}"
            ) from exc

        status = response.status_code
        if status in (401, 403):
            raise TranslationAuthError(
                f"{self.label}: authentication failed (HTTP {status}){_error_detail(response)}"
            )
        if status >= 400:
            raise TranslationError(
                f"{self.label}: HTTP {status}{_error_detail(response)}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise TranslationFormatError(
                f"{self.label}: reply was not JSON: {_excerpt(response.text)}"
            ) from exc
        if not isinstance(payload, dict):
            raise TranslationFormatError(
                f"{self.label}: reply was not a JSON object: {_excerpt(payload)}"
            )
        usage = payload.get("usage")
        self.last_usage = (
            {key: value for key, value in usage.items() if isinstance(value, int)}
            if isinstance(usage, dict)
            else {}
        )
        return payload

    # -- TranslationProvider ---------------------------------------------- #

    def translate_window(
        self,
        items: list[tuple[int, str]],
        *,
        source_language: str | None,
        target_language: str,
    ) -> dict[int, str]:
        if not items:
            return {}
        log.info(
            "%s: translating %d cues (%d chars, source=%s, target=%s)",
            self.label,
            len(items),
            sum(len(text) for _, text in items),
            source_language or "auto",
            target_language,
        )
        log.debug("%s: window payload: %s", self.label, items)
        try:
            payload = self._post(self._body(items, source_language, target_language))
            translations = _parse_translations(_message_content(payload))
        except TranslationError as exc:
            scrubbed = self._scrubbed(exc)
            if scrubbed is exc:
                raise
            raise scrubbed from exc
        log.info("%s: got %d translations, usage=%s", self.label, len(translations), self.last_usage)
        return translations

    def check_connection(self) -> str:
        """Round-trip one tiny cue so a bad key or model fails before a long job runs."""
        self.translate_window(
            [(1, "Hello there.")], source_language=None, target_language="English"
        )
        return f"{self.label} reachable"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def build_provider(
    settings: TranslationSettings,
    *,
    client: httpx.Client | None = None,
    api_key: str | None = None,
) -> OpenAICompatibleProvider:
    """Turn persisted settings into a provider, resolving preset and credentials.

    ``TranslationSettings.base_url``/``model`` win when set, so a user can point a
    preset at a proxy; the preset fills the gaps (and is the only source for
    ``custom``, which has none of its own).
    """
    provider = (settings.provider or "").strip()
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise TranslationError(f"unknown translation provider: {provider!r}")

    base_url = (settings.base_url or "").strip() or preset.base_url
    model = (settings.model or "").strip() or preset.model
    if not base_url or not model:
        raise TranslationError(
            f"provider {provider!r} needs both a base_url and a model; "
            "choose a preset or fill them in the translation settings"
        )

    key = api_key if api_key is not None else config.get_api_key(provider)
    if preset.needs_key and not key:
        names = ", ".join(config.env_var_names(provider))
        raise TranslationAuthError(
            f"{provider}: no API key found; store one in the settings or set {names}"
        )
    if key:
        logging_setup.register_secret(key)

    return OpenAICompatibleProvider(
        base_url=base_url,
        model=model,
        api_key=key,
        temperature=settings.temperature,
        timeout_s=settings.timeout_s,
        extra_instructions=settings.extra_instructions,
        client=client,
        provider_name=provider,
    )
