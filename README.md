# DoSubs

Windows desktop app that transcribes the spoken dialogue of video files, translates it
(Simplified Chinese by default) and exports three synchronised subtitle files.

**Download:** grab the latest packaged build from
[Releases](https://github.com/lewdoherz/dosubs/releases) — unzip `DoSubs-<version>-windows-x64.zip`,
optionally unzip the CUDA archive into `_cuda` next to `DoSubs.exe` for GPU transcription, and run
`DoSubs.exe`. The rest of this file covers building from source, the command line, and how the
pipeline works.

For `ExampleVideo.mp4` it writes, next to the video or into a chosen folder:

| File | Contents |
| --- | --- |
| `ExampleVideo.original.srt` | the transcribed dialogue in the spoken language |
| `ExampleVideo.zh-CN.srt` | the translation |
| `ExampleVideo.bilingual.srt` | original on the first line, translation underneath |

All three files are rendered from one canonical timeline, so they cannot drift apart.

Speech recognition runs locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(Whisper via CTranslate2). Translation is pluggable: DeepSeek (`deepseek-flash`) by default,
any OpenAI-compatible endpoint, or a local Ollama server. No recognition data ever leaves the
machine unless you enable a cloud translation provider.

## Requirements

- Windows 10/11, Python 3.12 or newer (developed on 3.14).
- FFmpeg and ffprobe on `PATH` (`winget install Gyan.FFmpeg`), or placed in
  `%LOCALAPPDATA%\DoSubs\bin`. Only used for media probing and for generating sample media in
  tests: audio itself is decoded in-process by PyAV, so no temporary audio files are written.
- Optional: an NVIDIA GPU. CPU transcription works, but is roughly 10-25x slower.

## Install (from source)

```powershell
cd C:\develop\DoSubs
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[gui,cuda,dev]"
```

- `gui` installs PySide6 (LGPL) for the desktop interface.
- `cuda` installs the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels so CUDA works without a
  system CUDA toolkit.
- `dev` installs pytest and pytest-qt.

Check the installation:

```powershell
.\.venv\Scripts\python.exe -m dosubs --selfcheck          # environment, GPU, ffmpeg, config paths
.\.venv\Scripts\python.exe -m dosubs --selfcheck --deep   # additionally loads the model and runs 1 s of audio
```

### Two dependency pins that matter

- **PyAV is pinned to 18.1.0.** faster-whisper 1.2.1 calls `av.open(..., metadata_errors=...)`,
  which PyAV 19 removed; without the pin every transcription fails with
  `TypeError: open() got an unexpected keyword argument 'metadata_errors'`.
- **CTranslate2 needs its CUDA libraries to be findable.** On Windows, CTranslate2 adds only its
  own package directory to the DLL search path, so the CUDA DLLs from the `nvidia-*-cu12` wheels
  are invisible. Measured on this workstation: `os.add_dll_directory` alone fails with
  `Library cublas64_12.dll is not found or cannot be loaded`, while prepending those directories
  to `PATH` (or preloading the DLLs with `ctypes`) works. `dosubs.gpu.bootstrap()` does both, and
  `dosubs.cli` runs it before CTranslate2 is imported.

Whisper weights are downloaded on first use into the standard Hugging Face cache
(`%USERPROFILE%\.cache\huggingface`), so models you already have are reused rather than
downloaded twice. Recognition then works offline.

## Desktop app

```powershell
.\.venv\Scripts\python.exe -m dosubs.app     # or: .\.venv\Scripts\dosubs-gui.exe
```

Drag video files onto the window (or use *Add files* / *Add folder*), choose the recognition and
translation settings, then *Start*. Progress is reported per stage (media probe, recognition, cue
building, translation, export) with a cancel button that stops the run after the current step and
leaves already written files in place.

### Reviewing and editing subtitles

The window has two tabs. **Queue** holds the file list, settings, progress and log. **Subtitles**
is the editor: it opens the subtitles of the first job that finished, so you can fix a translation,
correct a transcription, nudge a timing by 100 ms, add or delete a cue, and see the bilingual
rendering of the selected cue before re-exporting.

* Re-export renders all three files again from the edited transcript, so the timeline stays shared
  between them; the written files are re-read and verified afterwards.
* A timecode that cannot be parsed is rejected as you type it and the previous value is restored.
  Shifting a cue refuses to cross its neighbours.
* A timeline with real problems (overlap, zero-length cue) is reported and only written if you
  confirm — nothing is silently corrupted.

The editor is a review-and-correct loop rather than a modal gate in front of the export: the run
writes its files as it goes, and the editor rewrites them afterwards. If you want to inspect before
anything is written, run with `--dry-run` first (which prints the intended file names and writes
nothing).

## Command line

```powershell
# Transcribe + translate to zh-CN, all three files
.\.venv\Scripts\python.exe -m dosubs "D:\videos\ExampleVideo.mp4"

# A whole folder, English forced, GPU, into a chosen output directory
.\.venv\Scripts\python.exe -m dosubs "D:\videos" --language en --device cuda -o "D:\subs"

# Original subtitles only, CPU, overwriting existing files
.\.venv\Scripts\python.exe -m dosubs "D:\videos\ExampleVideo.mkv" --cpu --no-translate --only original --overwrite overwrite

# What would be written, without running anything
.\.venv\Scripts\python.exe -m dosubs "D:\videos\ExampleVideo.mp4" --dry-run

# Credentials and connectivity
.\.venv\Scripts\python.exe -m dosubs --set-key deepseek      # stored via Windows DPAPI (keyring)
.\.venv\Scripts\python.exe -m dosubs --test-provider
.\.venv\Scripts\python.exe -m dosubs --clear-cache
```

Run `python -m dosubs --help` for the full option list (`--model`, `--compute-type`,
`--no-vad`, `--batch-size`, `--target-language`, `--provider`, `--base-url`, `--temperature`,
`--format`, `--only`, `--overwrite`, `--no-cache`, `--save-config`, `--verbose`).

Inputs: `.mp4 .mkv .avi .mov .webm .m4v .wmv .flv .ts .mpg .mpeg` and
`.mp3 .wav .flac .m4a .aac .ogg .opus .wma .m4b`. Folders are scanned one level deep.
Filenames with spaces, Chinese characters or other Unicode are supported.

## How it works

```
probe (ffprobe) -> recognise (faster-whisper + VAD) -> build cues -> translate (context windows) -> export + verify
```

- **Recognition.** Voice activity detection is on by default; `condition_on_previous_text` is off
  (it is the usual cause of repetition loops on long files), `hallucination_silence_threshold`
  skips silence-only stretches, and word-level timestamps are requested because they cost
  essentially nothing (measured: 1.8 s vs 2.0 s for a 411 s clip) and give the cue builder real
  split points. An automatic language-detection result below 70 % confidence is reported as a
  warning instead of being trusted silently.
- **Cue building.** Raw recogniser segments are not usable as subtitles (measured outputs included
  cues of 39 s, 68 s and 106 s). `dosubs.segment` applies per-language budgets: maximum and
  minimum duration, characters per line, lines per cue, minimum gap, and a reading-speed ceiling.
  Chinese uses its own profile (18 characters per line, 11 characters/second, no line starting
  with `。，！？`, embedded Latin words kept whole). When bilingual output is requested, cues are
  built to one line per language so a bilingual cue stays two lines tall.
- **Translation.** Cues are translated in context windows (about 1600 characters, never splitting a
  cue) so the model sees neighbouring dialogue. Replies must map 1:1 by cue id; missing, empty,
  duplicated or unexpected ids are rejected, failed windows are retried with backoff and then
  split in half down to single cues, and the original timings are verified to be byte-identical
  afterwards. DeepSeek's thinking mode is explicitly disabled (`{"thinking": {"type": "disabled"}}`)
  because it is on by default and ignores `temperature`.
- **Export and verification.** Each file is written atomically and then re-read: cue count and
  every start/end timestamp are compared against the transcript, and any mismatch or failed
  translation is reported instead of being counted as success. The original subtitles are kept even
  when translation fails.
- **Cache.** Raw recogniser output is cached in `%LOCALAPPDATA%\DoSubs\cache\transcripts`, keyed by
  media identity (path, size, mtime) plus every recognition parameter that changes content.
  Re-running with a different target language skips recognition entirely; any mismatch (changed
  file, different model, corrupt entry) is a plain miss, never a silent reuse. Disable with
  `--no-cache`, clear with `--clear-cache`.

## Configuration and credentials

- Settings: `%APPDATA%\DoSubs\config.json` (written atomically).
- Logs: `%LOCALAPPDATA%\DoSubs\logs\dosubs.log` (rotating, 3 x 2 MB).
- API keys: Windows credential store through `keyring` (DPAPI-backed), or the environment
  variables `DOSUBS_DEEPSEEK_API_KEY` / `DEEPSEEK_API_KEY` (similarly for other providers). Keys are
  never written to the config file and are redacted from all log output.

Providers: `deepseek` (`https://api.deepseek.com`, `deepseek-flash`), `openai`,
`ollama` (`http://127.0.0.1:11434/v1`, fully offline once the model is pulled) and `custom`
for any other OpenAI-compatible `/chat/completions` endpoint.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not integration"   # unit + component suite, no network
.\.venv\Scripts\python.exe -m pytest -m integration         # loads real models, needs ffmpeg

# The integration tests that need real speech:
$env:DOSUBS_TEST_MEDIA = "D:\videos\clip.mp4"
$env:DOSUBS_TEST_MODEL = "large-v3"      # optional, default tiny
.\.venv\Scripts\python.exe -m pytest -m integration
```

Test layers, so nothing overclaims:

| Layer | What runs |
| --- | --- |
| Unit / component | real segmenter, exporter, verifier, cache, config and naming; speech recognition and the translation API are replaced by fakes, translation HTTP is exercised through `httpx.MockTransport` |
| Integration (no media env) | model loading and inference on 1 s of audio, GPU/CPU device selection, cancellation |
| Integration with `DOSUBS_TEST_MEDIA` | the whole pipeline with the real recogniser and a real local HTTP endpoint, through the CLI-equivalent `Pipeline` and through the actual desktop window |

No test claims a real transcription, a real translation or GPU compatibility unless it actually ran
one.

## Project layout

```
src/dosubs/
  models.py          shared data contracts (settings, cues, transcript, jobs)
  config.py          paths, persisted settings, credential storage
  gpu.py             CUDA DLL bootstrap and device/compute-type selection
  logging_setup.py   logging with credential redaction
  media/probe.py     ffmpeg/ffprobe discovery, media probing, input expansion
  asr/engine.py      faster-whisper wrapper: model cache, VAD, progress, cancellation
  segment/           per-language profiles and cue building
  translate/         provider interface, OpenAI-compatible client, batching, validation
  subs/              timecodes, output naming, SRT/ASS/VTT rendering, verification
  pipeline/          orchestration and the transcript cache
  ui/                PySide6 desktop interface
  cli.py, app.py     entry points
tests/               pytest suite (unit, component, integration)
```

## Packaging

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[packaging]"
packaging\build.ps1                            # dist\DoSubs         (about 119 MB)
packaging\build.ps1 -IncludeCuda               # + dist\DoSubs-cuda  (about 2.0 GB)
packaging\build.ps1 -IncludeCuda -Installer    # + installer, needs Inno Setup (iscc.exe)
```

What the build produces:

| Artifact | Size (measured here) | Contents |
| --- | --- | --- |
| `dist\DoSubs\` | 335 MB | the application: PySide6, faster-whisper, CTranslate2, PyAV, onnxruntime (Silero VAD), httpx, pysubs2, keyring. UPX is off; torch, whisperx, matplotlib, scipy and the unused Qt modules are excluded |
| `dist\DoSubs-cuda\` | 2.0 GB | the optional NVIDIA CUDA runtime (cuBLAS, cuDNN, NVRTC) in the flat `<component>\bin` layout the app discovers |
| `dist\DoSubs-<version>-setup.exe` | 90 MB | per-user installer (no admin prompt, Start Menu shortcut, uninstaller). By default it expects the CUDA component as a separate download; `-IncludeCuda -Installer` bundles it into one ~1.3 GB installer instead |

Deliberately **not** bundled:

- **The CUDA runtime, by default.** It is 2 GB unpacked (a single cuDNN DLL is 517 MB) and a
  machine without an NVIDIA GPU never needs it. Installed as a component, the app finds it next to
  the executable; `DOSUBS_CUDA_DIRS` can point at another location instead.
- **Speech models.** They download on first use into the Hugging Face cache, so recognition works
  offline afterwards and the installer stays small.
- **FFmpeg.** Found on `PATH` or in `%LOCALAPPDATA%\DoSubs\bin`; only used for probing media.

Diagnosing a packaged install (a GUI build has no console, so the report goes to a file):

```powershell
& "$env:LOCALAPPDATA\Programs\DoSubs\DoSubs.exe" --selfcheck         # -> %LOCALAPPDATA%\DoSubs\logs\selfcheck.txt
& "$env:LOCALAPPDATA\Programs\DoSubs\DoSubs.exe" --selfcheck --deep  # also loads the model on the chosen device
```

The packaged executable is the GUI; the scriptable command line stays `dosubs` from the virtual
environment.

Verified on this machine, with the real bundle:

| Check | Result |
| --- | --- |
| `DoSubs.exe --selfcheck` | reports the bundled interpreter, the config/log paths and the ffmpeg on `PATH` |
| `--selfcheck --deep`, no CUDA component | detection reports the GPU, the first attempt fails with `Library cublas64_12.dll is not found or cannot be loaded`, the run **degrades to the CPU automatically** and completes (1 s of audio took 3.43 s) |
| `--selfcheck --deep`, with `_cuda` next to the executable | CUDA runtime registered, model loaded, inference on `cuda/float16` (1 s of audio took 0.24 s) |
| `DoSubs-0.1.0-setup.exe /VERYSILENT` | installs into `%LOCALAPPDATA%\Programs\DoSubs` with no admin prompt; the installed copy's `--selfcheck` exits 0 and reports its installed path; `/VERYSILENT` uninstall exits 0 and removes the folder |

Note when scripting the packaged app: it is a windowed executable, so PowerShell's `&` returns
before it finishes — use `Start-Process -Wait` (or run it from a shell that waits) when the exit code
or its side effect matters.

### Publishing a release

1. Tag and push: `git tag -a v0.1.0 -m "DoSubs 0.1.0"` then `git push origin v0.1.0`.
2. Create a GitHub release for the tag (or let the API create the tag from `main`) and attach:
   `DoSubs-<version>-setup.exe`, `DoSubs-<version>-windows-x64.zip`,
   `DoSubs-cuda-<version>-windows-x64.zip` and a generated `SHA256SUMS.txt`.
   Release assets must each stay under 2 GiB — the CUDA archive is 1.28 GB, which is why it is a
   separate asset instead of being folded into the installer.
3. Use the release body as the end-user quick start: what each asset is, the `_cuda` step, the
   FFmpeg requirement, the provider key, and how to verify the machine with `--selfcheck --deep`.

Two packaging traps are handled in the spec and worth keeping in mind when editing it: the frozen
entry point must import the package rather than run a module as a script, and faster-whisper's
Silero VAD model is package data that has to be collected explicitly or every VAD run dies with an
ONNXRuntime `NO_SUCHFILE` error.

## Current limitations

Deliberate deviations from a naive reading of the brief, each with a reason:

- **No separate audio-extraction stage or temporary audio files.** faster-whisper decodes the
  audio stream in-process through PyAV, so there is no extraction step to report and nothing to
  clean up. This also removes a whole class of "left behind a 2 GB wav" failures.
- **The editor reviews after a run instead of gating it.** The pipeline writes its files as it goes;
  the *Subtitles* tab then opens and can rewrite them. `--dry-run` precedes anything.
- **No video preview inside the app** (your choice at kickoff): Qt's multimedia backend cannot be
  trusted with every MKV/AV1 file, and a broken preview would be worse than none.
- **`word_timestamps` is left on** even though nothing downstream displays word timings: the cue
  builder uses them as split points, and they cost about 10 % of run time.

Known gaps:

- Recognition accuracy (WER) and translation quality have not been scored against a labelled
  reference; both need a real evaluation on your media. Throughput figures come from
  silence-heavy sample clips and will be lower on dense dialogue.
- `large-v3-turbo` is offered but has not been benchmarked on this machine.
- The DeepSeek API path is implemented against the documented contract and verified against a stub
  server over real HTTP; it has not been exercised with a live API key from this repository. Run
  `dosubs --test-provider` once a key is stored to confirm it.
- The packaged build is verified for startup, for model loading plus inference on both the GPU and
  the CPU fallback path, and for install/uninstall (`--selfcheck --deep`, see *Packaging*). A full
  multi-file GUI run inside the frozen bundle has not been performed; the pipeline itself is
  verified from the virtual environment.
- Faster-whisper's Silero VAD model is loaded from package data. Inside the frozen build it lives
  in `_internal\faster_whisper\assets`, which the spec collects explicitly.
