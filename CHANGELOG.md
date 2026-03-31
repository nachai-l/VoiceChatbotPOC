# Changelog

## [Unreleased]

### Fixed — Voice Pipeline End-to-End
- **VAD int16 normalization**: Gradio streams raw int16 audio (RMS ~250), but VAD
  threshold was calibrated for float32 `[−1, 1]`. Added normalization in `compute_rms()`
  so energy threshold (0.006) works correctly with real browser mic input.
- **Silence trigger tuning**: Reduced `SILENCE_CHUNKS_TO_TRIGGER` from 6 → 4 and
  `MIN_SPEECH_CHUNKS` from 2 → 1 so turn-taking feels responsive and short
  follow-up utterances ("yes", "ok") are not dropped.
- **Audio playback killed by re-renders**: Both `stream_audio_chunk` (VAD) and
  `poll_pending_result` (timer) were writing to the same `audio_output` component
  every 0.25 s. Rapid `gr.update()` calls caused the audio player to re-render
  before it could start playback. Fixed by separating concerns: stream handler
  outputs 5 values (no audio), poller exclusively owns `audio_output`.
- **Audio output format**: Switched from WAV-file-path (`type="filepath"`) to
  numpy tuple (`type="numpy"`) — simpler and avoids Gradio file-serving issues
  inside Docker.
- **Live API send method**: Changed from `send_client_content` to
  `send_realtime_input` with `audio_stream_end=True`, matching the official
  Gemini Live API usage for audio input.
- **Docker API key**: Added `env_file` with `required: false` to docker-compose
  so `.env` is auto-loaded; also passes `GEMINI_API_KEY` from host environment.
- **Docker build speed**: Added `.dockerignore` to exclude `.venv`, `.git`,
  `__pycache__`, `tests` — reduced build context from ~480 MB to ~3 KB.

### Added
- **Retry logic** in `live_session_manager`: configurable timeout (60 s), one
  retry on completely empty responses, preservation of partial audio/transcripts
  on timeout or exception.
- **Timer-based poller** (`gr.Timer`, 0.25 s) to surface completed API tasks
  even when the mic stream callback stops firing.
- **Playback cooldown** (0.75 s) after assistant audio is returned, to prevent
  the always-on mic from immediately re-triggering on residual noise.
- **Diagnostic trace panel** showing live RMS, speech/silence chunk counts, and
  API result details.
- **118 unit tests** (up from initial 36) covering:
  - `_build_audio_output` numpy tuple creation and gain
  - `_consume_finished_task` success/error/cooldown paths
  - `poll_pending_result` pending/done/cooldown states
  - `stream_audio_chunk` 5-output contract (no audio)
  - Live API retry logic, timeout handling, `_has_meaningful_output`
  - `_build_live_config` structure
  - VAD constant sanity checks and `pending_task` lifecycle

### Known Issues
- **Stuck "Playing response..." state**: After the first successful turn, the UI
  can remain in "Playing response..." and not accept a second voice input. Root
  cause: the playback cooldown or poller does not reliably transition back to
  "Listening..." when the audio finishes. Needs investigation — likely requires
  either a longer cooldown calibrated to actual audio duration, or a client-side
  event (Gradio `audio_output.stop()`) to signal playback completion.
- **Short audio timeouts**: Very short utterances (< 1 s) sometimes produce
  empty API responses that time out at 60 s. The retry helps but does not fully
  resolve the issue.

---

## [0.1.0] — 2026-03-31

### Added — Phase 1 POC Skeleton
- Gradio 6.x UI with always-on microphone and energy-based VAD
- Gemini Live API integration (`gemini-3.1-flash-live-preview`)
- PCM audio codec (16-bit, 16 kHz input / 24 kHz output)
- Transcript handler with chat history
- Docker Compose local deployment
- 36 initial unit tests
