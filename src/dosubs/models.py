"""Shared data contracts for the DoSubs pipeline.

``Transcript`` is the single source of truth for timing: speech recognition fixes
the timeline, translation may only fill in ``Cue.text_translated``, and every
exporter renders from the same object. That is what keeps the original,
translated and bilingual subtitle files synchronised by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

__all__ = [
    "Stage",
    "MediaInfo",
    "Word",
    "AsrSegment",
    "Cue",
    "Transcript",
    "ProgressEvent",
    "OverwritePolicy",
    "ExportKind",
    "ExportTarget",
    "ExportPlan",
    "AsrSettings",
    "TranslationSettings",
    "ExportSettings",
    "JobSpec",
    "JobStatus",
    "JobResult",
    "SubtitleFormat",
    "PROVIDER_PRESETS",
    "ProviderPreset",
    "translation_settings_for",
]


class Stage(str, Enum):
    """Processing stages reported to the UI and the log.

    There is no separate audio extraction stage: faster-whisper decodes the audio
    stream in-process through PyAV, which avoids temporary files and a second
    decode pass.
    """

    PROBE = "probe"
    TRANSCRIBE = "transcribe"
    SEGMENT = "segment"
    TRANSLATE = "translate"
    EXPORT = "export"

    @property
    def label(self) -> str:
        return {
            Stage.PROBE: "Reading media information",
            Stage.TRANSCRIBE: "Recognising speech",
            Stage.SEGMENT: "Building subtitle cues",
            Stage.TRANSLATE: "Translating subtitles",
            Stage.EXPORT: "Writing subtitle files",
        }[self]


class SubtitleFormat(str, Enum):
    SRT = "srt"
    ASS = "ass"
    VTT = "vtt"


class ExportKind(str, Enum):
    ORIGINAL = "original"
    TRANSLATED = "translated"
    BILINGUAL = "bilingual"


class OverwritePolicy(str, Enum):
    """What to do when a subtitle file already exists."""

    ASK = "ask"
    OVERWRITE = "overwrite"
    SKIP = "skip"
    RENAME = "rename"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """What we learned about an input file from ffprobe."""

    path: Path
    duration_s: float
    size_bytes: int
    has_audio: bool
    audio_codec: str | None = None
    audio_channels: int | None = None
    audio_sample_rate: int | None = None
    video_codec: str | None = None
    container: str | None = None
    title: str | None = None

    @property
    def has_video(self) -> bool:
        return self.video_codec is not None


@dataclass(frozen=True, slots=True)
class Word:
    """A single recogniser word with its own timing."""

    start_ms: int
    end_ms: int
    text: str
    probability: float = 1.0


@dataclass(frozen=True, slots=True)
class AsrSegment:
    """One raw recogniser segment, independent of the engine that produced it.

    The speech engine emits these; :mod:`dosubs.segment.cue_builder` turns them
    into subtitle cues.
    """

    start_ms: int
    end_ms: int
    text: str
    words: tuple[Word, ...] = ()
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


@dataclass(slots=True)
class Cue:
    """One subtitle cue. Timings are fixed by the recogniser/segmenter."""

    id: int
    start_ms: int
    end_ms: int
    text_original: str
    text_translated: str = ""
    words: tuple[Word, ...] = ()

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def is_translated(self) -> bool:
        return bool(self.text_translated.strip())

    def copy(self, **changes: object) -> "Cue":
        return replace(self, **changes)  # type: ignore[arg-type]


@dataclass(slots=True)
class Transcript:
    """Canonical subtitle timeline for one media file."""

    media: MediaInfo
    cues: list[Cue] = field(default_factory=list)
    source_language: str | None = None
    language_probability: float | None = None
    language_forced: bool = False
    asr_params: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cues)

    def renumber(self) -> None:
        """Force cue ids to 1..N in timeline order."""
        self.cues.sort(key=lambda c: (c.start_ms, c.end_ms))
        for index, cue in enumerate(self.cues, start=1):
            cue.id = index

    def has_translations(self) -> bool:
        return any(cue.is_translated for cue in self.cues)

    def untranslated_ids(self) -> list[int]:
        return [cue.id for cue in self.cues if not cue.is_translated]

    def validate_timeline(
        self, *, allow_overlap: bool = False, min_duration_ms: int = 0
    ) -> list[str]:
        """Return human-readable problems; empty list means the timeline is sound."""
        problems: list[str] = []
        previous: Cue | None = None
        for cue in self.cues:
            if cue.start_ms < 0 or cue.end_ms < 0:
                problems.append(f"cue {cue.id}: negative timestamp")
            if cue.end_ms <= cue.start_ms:
                problems.append(
                    f"cue {cue.id}: end ({cue.end_ms} ms) is not after start ({cue.start_ms} ms)"
                )
            if min_duration_ms and 0 < cue.duration_ms < min_duration_ms:
                problems.append(
                    f"cue {cue.id}: duration {cue.duration_ms} ms is below {min_duration_ms} ms"
                )
            if previous is not None and cue.start_ms < previous.end_ms and not allow_overlap:
                problems.append(
                    f"cue {cue.id}: starts at {cue.start_ms} ms, overlapping cue "
                    f"{previous.id} which ends at {previous.end_ms} ms"
                )
            previous = cue
        return problems


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """A single progress report from the pipeline."""

    stage: Stage
    message: str
    fraction: float | None = None
    file_index: int = 1
    file_total: int = 1

    @property
    def percent(self) -> float | None:
        return None if self.fraction is None else round(self.fraction * 100, 1)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AsrSettings:
    """Speech recognition configuration."""

    model: str = "large-v3"
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "auto"  # auto | float16 | int8_float16 | int8 | float32
    language: str | None = None  # None = detect automatically
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    hallucination_silence_threshold: float | None = 2.0
    initial_prompt: str | None = None
    batch_size: int = 0  # 0 = sequential; >1 uses the batched pipeline

    def cache_key(self) -> dict[str, object]:
        """Parameters that change the transcript content (used for cache identity)."""
        return {
            "model": self.model,
            "language": self.language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": self.condition_on_previous_text,
            "no_speech_threshold": self.no_speech_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
            "initial_prompt": self.initial_prompt,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    """A named OpenAI-compatible endpoint."""

    name: str
    base_url: str
    model: str
    needs_key: bool
    note: str


#: Presets for the OpenAI-compatible chat-completions client. Any other
#: endpoint can be reached by choosing ``custom`` and setting base_url/model.
PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        name="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        needs_key=True,
        note="DeepSeek: best quality/price for en->zh. Off-peak is half price.",
    ),
    "openai": ProviderPreset(
        name="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-mini",
        needs_key=True,
        note="OpenAI-hosted models.",
    ),
    "ollama": ProviderPreset(
        name="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3:8b",
        needs_key=False,
        note="Local Ollama server; fully offline, needs the model pulled first.",
    ),
    "custom": ProviderPreset(
        name="custom",
        base_url="",
        model="",
        needs_key=False,
        note="Any OpenAI-compatible /chat/completions endpoint.",
    ),
}


@dataclass(slots=True)
class TranslationSettings:
    """Machine translation configuration.

    Prefer :func:`translation_settings_for` when switching providers: the
    dataclass defaults describe DeepSeek, so changing ``provider`` alone would
    leave the endpoint and model pointing at DeepSeek.
    """

    enabled: bool = True
    provider: str = "deepseek"
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    target_language: str = "zh-CN"
    temperature: float = 0.2
    #: Cues are translated in context windows of at most this many characters.
    window_chars: int = 1600
    #: Characters above which a single window is split further on retry.
    max_retries: int = 4
    timeout_s: float = 120.0
    extra_instructions: str = ""


def translation_settings_for(provider: str, **overrides: object) -> TranslationSettings:
    """Build settings for a named provider preset, then apply overrides.

    ``overrides`` may pass any :class:`TranslationSettings` field, including
    ``model``/``base_url`` to depart from the preset.
    """
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ValueError(
            f"unknown translation provider {provider!r}; known: {', '.join(sorted(PROVIDER_PRESETS))}"
        )
    values: dict[str, object] = {
        "provider": preset.name,
        "model": preset.model,
        "base_url": preset.base_url,
    }
    values.update(overrides)
    return TranslationSettings(**values)  # type: ignore[arg-type]


@dataclass(slots=True)
class ExportSettings:
    """Output selection and file policy."""

    output_dir: Path | None = None  # None = next to each video
    format: SubtitleFormat = SubtitleFormat.SRT
    write_original: bool = True
    write_translated: bool = True
    write_bilingual: bool = True
    overwrite: OverwritePolicy = OverwritePolicy.ASK
    encoding: str = "utf-8"
    bom: bool = False

    def wanted_kinds(self) -> list[ExportKind]:
        kinds: list[ExportKind] = []
        if self.write_original:
            kinds.append(ExportKind.ORIGINAL)
        if self.write_translated:
            kinds.append(ExportKind.TRANSLATED)
        if self.write_bilingual:
            kinds.append(ExportKind.BILINGUAL)
        return kinds


@dataclass(slots=True)
class JobSpec:
    """One media file to process."""

    media: Path
    asr: AsrSettings
    translation: TranslationSettings
    export: ExportSettings


@dataclass(frozen=True, slots=True)
class ExportTarget:
    kind: ExportKind
    path: Path


@dataclass(frozen=True, slots=True)
class ExportPlan:
    """Resolved output files for one job."""

    targets: tuple[ExportTarget, ...] = ()
    conflicts: tuple[Path, ...] = ()  # existing files that need a decision (ASK policy)
    skipped: tuple[ExportTarget, ...] = ()  # existing files the policy told us to leave alone

    def paths(self) -> list[Path]:
        return [target.path for target in self.targets]


@dataclass(slots=True)
class JobResult:
    """Outcome of one job."""

    media: Path
    status: JobStatus
    written: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    error: str | None = None
    transcript_path: Path | None = None
