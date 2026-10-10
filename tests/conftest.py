"""Shared test bootstrap and fixtures.

``src`` goes first on ``sys.path`` so the suite runs whether or not the project has
been installed into the virtual environment. Media is generated with ffmpeg rather
than committed, so the tests stay small and honest about what ffmpeg produces.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def have_ffmpeg() -> bool:
    """True when both ffmpeg and ffprobe are usable."""
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _generate_video(target: Path, seconds: int) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440",
            "-t",
            str(seconds),
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(target),
        ],
        check=True,
        creationflags=_CREATE_NO_WINDOW,
    )
    return target


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep transcript-cache writes out of the developer's real %LOCALAPPDATA%.

    Without this, one test's cached recognition result changes the behaviour of the
    next test that processes the same media file, which makes the suite order
    dependent. Tests that exercise the cache pass ``cache_root`` explicitly.
    """
    root = tmp_path / "cache"
    monkeypatch.setattr("dosubs.pipeline.cache.cache_dir", lambda: root)
    return root


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A short real video with an audio stream, generated once per session."""
    if not have_ffmpeg():
        pytest.skip("ffmpeg is required to generate sample media")
    return _generate_video(tmp_path_factory.mktemp("media") / "ExampleVideo.mp4", 8)


@pytest.fixture()
def make_video_file(tmp_path: Path) -> Callable[..., Path]:
    """Factory for extra media files (unicode names, audio-only, corrupt, ...)."""

    def _make(
        name: str = "ExampleVideo.mp4", seconds: int = 8, directory: Path | None = None
    ) -> Path:
        if not have_ffmpeg():
            pytest.skip("ffmpeg is required to generate sample media")
        return _generate_video((directory or tmp_path) / name, seconds)

    return _make


# --------------------------------------------------------------------------- #
# Local translation endpoint
# --------------------------------------------------------------------------- #

#: Translation text every stub reply contains, so tests can assert it round-trips.
STUB_TRANSLATION = "这是测试用的中文译文。"


class _StubTranslationHandler(BaseHTTPRequestHandler):
    """Answers ``/chat/completions`` with a fixed Chinese line per requested cue id."""

    protocol_version = "HTTP/1.1"
    seen_requests = 0

    def log_message(self, *args) -> None:  # keep pytest output clean
        pass

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).seen_requests += 1

        cues: list[dict] = []
        for message in body.get("messages", []):
            if message.get("role") == "user":
                try:
                    cues.extend(json.loads(message["content"]).get("cues", []))
                except (TypeError, ValueError):
                    pass

        reply = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "translations": [
                                    {"id": cue["id"], "text": f"{STUB_TRANSLATION}[{cue['id']}]"}
                                    for cue in cues
                                ]
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }
        payload = json.dumps(reply, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="session")
def stub_server() -> str:
    """Base URL of an OpenAI-compatible endpoint served in-process."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubTranslationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="session")
def real_media() -> Path:
    """A real video file supplied through ``DOSUBS_TEST_MEDIA``."""
    raw = os.environ.get("DOSUBS_TEST_MEDIA")
    if not raw:
        pytest.skip("set DOSUBS_TEST_MEDIA to a video or audio file to run this test")
    path = Path(raw)
    if not path.exists():
        pytest.skip(f"DOSUBS_TEST_MEDIA points at a missing file: {path}")
    return path
