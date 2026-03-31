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

### Lessons Learned — Root Causes of Major Bugs

1. **Never assume the audio format between framework boundaries.**
   Gradio's `gr.Audio(type="numpy", streaming=True)` sends raw int16 samples
   (values up to ±32767), not normalized float32 `[−1, 1]`. The VAD threshold
   of 0.01 was meaningless against RMS values of 250+. The silence detector
   literally never fired — every chunk looked like speech.
   *Takeaway*: When two systems exchange data (Gradio → VAD → API), log the
   actual values at each boundary on the first run. A single `print(rms)` would
   have caught this immediately.

2. **Two handlers writing to one Gradio component = invisible data race.**
   `stream_audio_chunk` (mic callback, every 0.25 s) and `poll_pending_result`
   (timer, every 0.25 s) both output to `audio_output`. Even though
   `gr.update()` is documented as a no-op, every yield/return triggers a
   component re-render in the browser. The audio player was being destroyed and
   recreated 4–8 times per second — it never got a chance to start playback.
   The symptom (audio waveform visible but 0:00 duration, no sound) gave no
   hint that the root cause was a render storm.
   *Takeaway*: In Gradio, each output component should have exactly one handler
   that writes to it. If multiple event sources need to update the same
   component, funnel them through a single handler (the poller pattern used
   here).

3. **Wrong API method fails silently with WebSocket streaming.**
   `send_client_content()` is for text turns. For audio, the Gemini Live API
   requires `send_realtime_input(audio=Blob(...))` + `audio_stream_end=True`.
   Using the wrong method didn't raise an error — the WebSocket just hung
   waiting for `turn_complete` that never came, until the 60 s timeout.
   *Takeaway*: Always cross-reference with the official SDK examples for the
   exact method signatures. "Works in the text demo" does not mean it works for
   audio.

4. **Docker volume mount masks the image; env vars don't persist across shells.**
   `volumes: [".:/app"]` means the container runs host code, not the built
   image. Combined with `GEMINI_API_KEY=${GEMINI_API_KEY:-}` in docker-compose,
   the key is only available if exported in the *same terminal session* that
   runs `docker compose up`. Switching terminals or restarting the shell loses
   the key with no error — the container starts fine but the API call fails
   with "GEMINI_API_KEY not set".
   *Takeaway*: For secrets in local Docker dev, prefer a `.env` file (committed
   to `.gitignore`) over shell exports. It survives terminal restarts and is
   visible in the project.

5. **Gradio 6.x breaking changes are not caught by unit tests alone.**
   `gr.Chatbot(type="messages")`, `show_copy_button`, and `theme=` in
   `gr.Blocks()` all worked in Gradio 5.x but threw `TypeError` in 6.x. The
   Python test suite passed locally (where Gradio 5.x was installed) but the
   Docker image had Gradio 6.x. These are constructor-time errors that only
   surface when `create_app()` is actually called.
   *Takeaway*: Pin the Gradio version in `requirements.txt` and run
   `create_app()` in at least one test to catch constructor regressions early.

---

## [0.1.0] — 2026-03-31

### Added — Phase 1 POC Skeleton
- Gradio 6.x UI with always-on microphone and energy-based VAD
- Gemini Live API integration (`gemini-3.1-flash-live-preview`)
- PCM audio codec (16-bit, 16 kHz input / 24 kHz output)
- Transcript handler with chat history
- Docker Compose local deployment
- 36 initial unit tests
