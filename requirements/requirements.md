# POC Requirements — Real-Time Voice Agent (Gemini Live + Flash-Lite Orchestration)

**Document status:** Draft
**Last updated:** 2026-03-31
**Model IDs last verified:** 2026-03-31 — re-verify before implementation begins

---

## Table of Contents

1. [Document Purpose](#1-document-purpose)
2. [POC Objective](#2-poc-objective)
3. [Recommended POC UI Choice](#3-recommended-poc-ui-choice)
4. [Target POC Scope](#4-target-poc-scope)
5. [High-Level Architecture](#5-high-level-architecture)
6. [Required Models and Platform Components](#6-required-models-and-platform-components)
7. [Functional Requirements](#7-functional-requirements)
8. [Orchestration Requirements for Flash-Lite](#8-orchestration-requirements-for-flash-lite)
9. [Session and Memory Requirements](#9-session-and-memory-requirements)
10. [UI Requirements](#10-ui-requirements)
11. [Non-Functional Requirements](#11-non-functional-requirements)
12. [Suggested Tooling and Backend Modules](#12-suggested-tooling-and-backend-modules)
13. [Acceptance Criteria](#13-acceptance-criteria)
14. [Demo Scenarios](#14-demo-scenarios)
15. [Recommended Delivery Plan](#15-recommended-delivery-plan)
16. [Final Recommendation](#16-final-recommendation)

---

## 1. Document Purpose

This document defines the requirements for a Proof of Concept (POC) of a real-time voice AI assistant using **`gemini-3.1-flash-live-preview`** as the conversational voice layer and **`gemini-3.1-flash-lite-preview`** as the orchestration layer. The POC will expose a browser-based GUI using **Gradio** or **Streamlit**, support natural spoken interaction, and execute **1–2 business tools** through a controlled backend orchestration flow.

The core idea is:

- **Gemini 3.1 Flash Live Preview** handles the live spoken conversation
- **Gemini 3.1 Flash-Lite Preview** handles text-based orchestration and deterministic workflow decisions
- The backend executes business tools and returns results to the live session for spoken delivery to the user

This split is aligned with current Google model capabilities: Flash Live is a low-latency audio-to-audio model with Live API and function calling support, while Flash-Lite is a cost-efficient text-output model suited for high-volume agentic tasks and supports structured outputs, caching, code execution, file search, function calling, and URL context. ([Google AI for Developers][1])

---

## 2. POC Objective

The POC must demonstrate that a voice-first assistant can:

1. Hold a natural voice conversation with the user in near real time
2. Accept spoken input and return spoken output
3. Handle interruption during model speech
4. Route selected user intents into a backend orchestration layer
5. Use Flash-Lite to decide whether and how to call tools
6. Execute **at least 2 tools**
7. Return tool results as concise, natural spoken responses
8. Show enough system traceability to explain what happened during the conversation

The POC is intended to validate:

- Feasibility of the architecture
- Quality of the user experience
- Latency acceptability
- Tool-routing correctness
- Extensibility toward a production voice agent

The Live API is designed for low-latency voice interactions, supports barge-in, tool use, audio transcriptions, and stateful WebSocket sessions. ([Google AI for Developers][2])

---

## 3. Recommended POC UI Choice

### Preferred option: Gradio

For this specific POC, **Gradio is the recommended GUI**.

Why:

- Gradio has built-in **audio input/output** components
- Gradio has a **Chatbot** component that can display markdown and media
- Gradio has official guidance for **real-time speech recognition** demos
- It is generally better suited for a quick, demo-friendly, multimodal interface

Gradio officially supports recording audio input, playing audio output, and building chat-style interfaces, and its guides include real-time speech workflows. ([gradio.app][3])

### Acceptable alternative: Streamlit

Streamlit is acceptable if the team is more comfortable with it, especially for a **push-to-talk, turn-based** demo. Streamlit provides `st.audio_input` for microphone recording and built-in chat elements for conversational UIs. However, for a voice-first POC with richer audio UX, Gradio is the better fit. ([Streamlit Docs][4])

---

## 4. Target POC Scope

### In scope

- Browser-based demo UI in **Gradio** or **Streamlit**
- Real-time or near-real-time voice conversation
- Live connection to **`gemini-3.1-flash-live-preview`**
- Backend orchestration using **`gemini-3.1-flash-lite-preview`**
- At least **2 callable tools**
- Text transcript display for user input and assistant output
- Visible trace/log panel for:
  - recognized intent
  - orchestration decision
  - tool invoked
  - tool result
  - final user-facing response
- Basic session continuity during the demo
- Basic guardrails and error handling

### Out of scope

- Production-grade authentication and IAM
- Full banking-grade fraud stack
- Omnichannel integration
- Telephony integration
- Multilingual production tuning
- Production observability platform
- Persistent user profile platform
- Fully automated compliance controls
- Production RAG or enterprise knowledge ingestion platform

---

## 5. High-Level Architecture

### 5.1 Logical architecture

```text
Browser GUI (Gradio preferred / Streamlit acceptable)
  ├─ microphone input
  ├─ transcript panel
  ├─ response audio playback
  └─ debug / tool trace panel

Python App Server
  ├─ GUI server
  ├─ Live session manager
  ├─ audio conversion / buffering
  ├─ prompt and session-state manager
  ├─ orchestrator bridge
  └─ tool execution layer

Gemini 3.1 Flash Live Preview
  ├─ voice conversation
  ├─ interruption handling
  ├─ live spoken response
  └─ single backend function call: run_orchestrator(...)

Gemini 3.1 Flash-Lite Preview
  ├─ intent interpretation
  ├─ deterministic workflow classification
  ├─ structured orchestration output
  ├─ tool selection
  └─ response planning

Backend Tools
  ├─ Tool 1: Balance Inquiry
  └─ Tool 2: Transaction Status Check
```

### 5.2 Architectural principle

The Live model should remain the **conversation layer**, not the full business brain.

The Flash-Lite layer should remain the **decision/orchestration layer**, where it classifies the request, determines whether tool use is needed, chooses the correct tool, validates parameters, and returns a structured action plan for the backend to execute.

This design is strongly preferred because Live supports function calling but only **synchronous** function calling for Gemini 3.1 Flash Live, and tool responses must be handled manually by the client/backend. Flash-Lite, by contrast, supports structured outputs and richer agentic features. ([Google AI for Developers][5])

---

## 6. Required Models and Platform Components

> **Note:** Model IDs for preview models can expire or be renamed. Verify each model ID against the [Google AI developer console](https://ai.google.dev/gemini-api/docs/models) before implementation begins. Last verified: **2026-03-31**.

### 6.1 Live conversation model

**Model:** `gemini-3.1-flash-live-preview`

| Capability | Supported |
|---|---|
| Inputs | text, image, audio, video |
| Outputs | text and audio |
| Audio generation | Yes |
| Live API | Yes |
| Function calling | Yes |
| Structured outputs | No |
| File search | No |
| URL context | No |
| Caching | No |

([Google AI for Developers][1])

### 6.2 Orchestration model

**Model:** `gemini-3.1-flash-lite-preview`

| Capability | Supported |
|---|---|
| Output | text only |
| Function calling | Yes |
| Structured outputs | Yes |
| Caching | Yes |
| Code execution | Yes |
| File search | Yes |
| URL context | Yes |
| Live API | No |

([Google AI for Developers][6])

### 6.3 Connection pattern

For this POC, use **server-to-server Live API integration** from the Python app server. This is the simplest and safest architecture for a Gradio or Streamlit prototype.

Google documents two main approaches for Live API integration:

- **server-to-server**: the backend connects to Live API over WebSockets
- **client-to-server**: the frontend connects directly using WebSockets and ephemeral tokens

For this POC, direct browser connection is not required. Ephemeral tokens should be treated as a future hardening step if the architecture later moves to direct client connection. ([Google AI for Developers][2])

---

## 7. Functional Requirements

### FR-1. Voice input

The user must be able to speak into the GUI using the browser microphone.

The system must accept spoken input, package it into the format required by the Live API, and forward it to the live session.

**Audio format:**
- Input: 16-bit PCM, 16kHz, little-endian
- Output: 16-bit PCM, 24kHz, little-endian

Both formats must be handled by the `audio_codec.py` module (see §12). ([Google AI for Developers][2])

### FR-2. Voice output

The system must return spoken responses from the assistant and play them in the GUI.

### FR-3. Text transcript

The system must display:

- User transcript
- Assistant transcript
- Current tool/action state

The Live API supports audio transcriptions for both user input and model output. ([Google AI for Developers][2])

### FR-4. Interruption / barge-in

The user must be able to interrupt the assistant while it is speaking, and the assistant must stop or adapt appropriately.

Barge-in is a documented Live API feature. ([Google AI for Developers][2])

### FR-5. Business tool invocation through Flash-Lite

When the user asks for a supported operational task, the system must:

1. Capture the user speech
2. Let Flash Live handle the conversation turn
3. Call a single backend orchestration function such as `run_orchestrator`
4. Pass a normalized text request and session state into Flash-Lite
5. Let Flash-Lite return a structured orchestration decision
6. Execute the selected tool
7. Summarize the result for user delivery
8. Send the final response back to Flash Live for spoken output

### FR-6. Minimum tools

The POC must implement **at least 2 tools**.

#### Tool A: Balance Inquiry

Suggested signature:

- `get_balance(customer_id, auth_token | last4_phone | dob)`

> **Note:** For the POC, authentication parameters are simulated. The test fixture should accept a hardcoded `customer_id` and one of the auth fields without real credential validation.

Expected response:

- Account type
- Current balance
- Available balance
- Timestamp
- Status

#### Tool B: Transaction Status Check

Suggested signature:

- `get_transaction_status(reference_id, customer_id)`

Expected response:

- Transaction reference
- Transaction type
- Current status
- Last update timestamp
- Next action or note

These tools may be mock APIs or simulated backend services for the POC.

### FR-7. Tool response explainability

The debug panel must show:

- Recognized intent
- Whether Flash-Lite selected a tool
- Selected tool name
- Parameters passed
- Tool result summary
- Final spoken response

### FR-8. Fallback response path

If the request does not require a tool, Flash-Lite may classify the request as:

- Direct conversational answer
- Unsupported request
- Clarification required
- Escalation required

### FR-9. Clarification handling

If Flash-Lite returns a non-empty `missing_fields` list, the orchestration loop must **not** call the tool. Instead, the backend must format the first missing field into a natural follow-up question and route it back to Flash Live for spoken delivery.

The loop must retry orchestration once the user supplies the missing value.

Examples:

- "Please tell me your transaction reference number."
- "Please confirm your birth date."

### FR-10. Error handling

If a tool fails, the system must:

- Avoid hallucinating a successful result
- Tell the user that the action could not be completed
- Provide a fallback or retry message
- Log the failure in the trace panel

---

## 8. Orchestration Requirements for Flash-Lite

Flash-Lite must be used as a **dedicated orchestration model**, not as the live voice model.

### 8.1 Input to Flash-Lite

Each orchestration call must include:

- Normalized user utterance text
- Current session summary
- Selected business domain
- Allowed tools list
- Tool schemas
- Safety / policy notes
- Required structured output schema

### 8.2 Output from Flash-Lite

Flash-Lite must return structured JSON conforming to this schema:

```json
{
  "intent": "balance_inquiry | transaction_status | faq | clarify | unsupported",
  "requires_tool": true,
  "selected_tool": "get_balance",
  "confidence": 0.92,
  "missing_fields": [],
  "tool_arguments": {
    "customer_id": "C12345"
  },
  "user_message_before_tool": "Let me check that for you.",
  "user_message_after_tool": "Your available balance is ...",
  "reason": "User explicitly asked for account balance"
}
```

**`missing_fields` handling:** If `missing_fields` is non-empty, the backend must not call any tool. It must use `user_message_before_tool` as the follow-up prompt and re-invoke Flash-Lite after receiving the user's answer (see FR-9).

This requirement is appropriate because Flash-Lite supports structured outputs and function calling, whereas Flash Live does not support structured outputs. ([Google AI for Developers][6])

### 8.3 Orchestration constraints

Flash-Lite must be instructed to:

- Choose only from the allowed tool list
- Never invent tools
- Never claim tool completion before receiving tool output
- Ask for clarification when required arguments are missing
- Return only valid schema-conforming JSON
- Prefer deterministic workflows over open-ended reasoning for supported tool intents

---

## 9. Session and Memory Requirements

### SR-1. Live session management

The system must manage Live API session lifetime carefully.

Google documents that:

- Without compression, **audio-only sessions are limited to 15 minutes**
- Connection lifetime is around **10 minutes**
- Context window compression can extend sessions to an unlimited amount of time
- Session resumption can keep a session alive across multiple connections

([Google AI for Developers][7])

**Reconnection behavior:** The backend must detect connection drops (WebSocket close events or timeout) and attempt to resume the session using the Live API session resumption mechanism. During reconnection, the UI must show a "reconnecting..." state indicator. If reconnection fails after one retry, the system must display an error and offer the user a "start new session" option. The backend session state (SR-2) must be preserved across reconnections so the conversation can continue without data loss.

### SR-2. POC memory strategy

The POC must not rely only on the raw Live session history.

It must maintain a lightweight backend session state containing:

- User name or pseudonymous session ID
- Important verified values
- Last successful tool result summary
- Current workflow step
- Latest unresolved question
- Short rolling conversation summary

### SR-3. Compression-aware design

If the POC enables context window compression, it must assume that older context may be truncated and should re-inject only the important state summary into subsequent orchestration calls. This is an implementation inference from the documented sliding-window compression behavior. ([Google AI for Developers][7])

---

## 10. UI Requirements

### 10.1 Required UI elements

The GUI must include:

- Title and environment label
- Microphone input control
- Conversation transcript panel
- Assistant state indicator:
  - Listening
  - Thinking
  - Checking tool
  - Speaking
  - Reconnecting
  - Error
- Tool trace/debug panel
- Clear session / reset button
- Optional latency metrics panel

### 10.2 Recommended Gradio layout

If using Gradio:

- `gr.Audio` for microphone input/output
- `gr.Chatbot` for transcript and response history
- `gr.Markdown` or `gr.JSON` for trace/debug output
- `gr.State` for in-session state
- `Blocks` layout for flexible composition

Gradio's official components support recorded audio input/output and chatbot rendering of media-rich chat content. ([gradio.app][3])

### 10.3 Acceptable Streamlit layout

If using Streamlit:

- `st.audio_input` for microphone capture
- Chat elements for transcript display
- Side panel for state / logs / debug
- Audio playback for model output

Streamlit officially provides `st.audio_input` and chat elements suitable for conversational apps. ([Streamlit Docs][4])

---

## 11. Non-Functional Requirements

### NFR-1. Latency target

For the POC, target:

- End of user utterance to first assistant response: **< 2.5 seconds**
- Tool-free turns: ideally **< 2.0 seconds**
- Tool-backed turns: ideally **< 4.0 seconds**

These are proposed engineering targets for the POC, not vendor guarantees.

### NFR-2. Stability target

The demo should run for at least:

- **20 minutes continuous usage**
- **10 consecutive successful tool-backed conversations**

### NFR-3. Observability

The system must log:

- Session ID
- Turn ID
- Timestamps
- Detected intent
- Flash-Lite output
- Selected tool
- Tool latency
- Final response latency
- Error events

### NFR-4. Security

For the POC:

- API keys must remain server-side
- No secrets in frontend code
- Tool credentials must be environment-configured
- PII in logs must be minimized or masked
- Recorded audio persistence must be disabled by default unless explicitly enabled for testing

### NFR-5. Safety

The system must block or safely deflect:

- Unsupported high-risk advice
- Sensitive actions without required confirmation
- Fabricated tool results
- Requests outside the allowed POC domain

---

## 12. Suggested Tooling and Backend Modules

The POC backend should include these modules:

```text
/app
  /ui
    gradio_app.py or streamlit_app.py
  /live
    live_session_manager.py       # WebSocket session lifecycle + reconnection logic
    audio_codec.py                # Input decode (PCM 16kHz) + output encode (PCM 24kHz)
    transcript_handler.py
  /orchestration
    flash_lite_orchestrator.py
    schemas.py
    prompts.py
  /tools
    balance_tool.py
    transaction_status_tool.py
  /state
    session_store.py
    summary_store.py
  /safety
    guardrails.py
  /logging
    telemetry.py
  /tests
    test_tools.py
    test_orchestrator.py
    test_e2e.py
```

---

## 13. Acceptance Criteria

The POC is accepted only if all of the following pass:

### Functional

| ID | Criterion |
|---|---|
| AC-1 | User can open the GUI, speak, and hear a spoken assistant response |
| AC-2 | User and assistant transcripts are visible in the UI |
| AC-3 | When the user asks for account balance, the balance tool is invoked and the result is returned verbally and in text |
| AC-4 | When the user asks for transaction status, the transaction-status tool is invoked and the result is returned verbally and in text |
| AC-5 | When required parameters are missing, the assistant asks a follow-up question instead of calling the tool incorrectly |
| AC-6 | The user can interrupt the assistant and the conversation continues correctly |
| AC-7 | If a tool returns an error, the assistant gives a safe fallback message and the error appears in the debug trace |
| AC-8 | The system retains enough state to complete a two-step tool interaction in one session |
| AC-9 | The UI displays the selected intent, chosen tool, and tool outcome for each tool-backed turn |

### Non-functional

| ID | Criterion |
|---|---|
| AC-10 | At least 3 consecutive tool-backed turns complete within the 4.0s latency target in a local demo environment |
| AC-11 | The demo runs for 20 minutes without a session crash or unrecoverable error |
| AC-12 | No API keys or credentials appear in browser devtools network requests or frontend source |

---

## 14. Demo Scenarios

The POC must support at least these demo scenarios:

1. **Greeting + help discovery**
   - "Hi, what can you help me with?"

2. **Balance inquiry**
   - "What is my savings account balance?"

3. **Transaction status**
   - "Can you check the status of my recent transfer?"

4. **Clarification flow**
   - User omits reference number → assistant asks for it → user provides it → tool executes

5. **Interruption flow**
   - User interrupts while assistant is speaking

6. **Fallback / unsupported**
   - User requests something outside the supported tool scope

7. **Error flow**
   - Tool timeout or simulated backend failure

---

## 15. Recommended Delivery Plan

### Phase 1 — Skeleton

- Project scaffolding
- GUI shell
- Live API connection
- Microphone input and audio playback
- Transcript display

### Phase 2 — Orchestration

- Flash-Lite structured orchestration prompt
- Orchestration schema validation
- `missing_fields` clarification loop
- Session state handling
- Debug trace panel

### Phase 3 — Tools

- Implement balance inquiry tool (mock)
- Implement transaction status tool (mock)
- Wire tool execution and response loop

### Phase 4 — Polish

- Reconnection handling
- Interruption handling tests
- Fallback messages
- Latency measurement
- Demo script and sample data
- README and run guide

---

## 16. Final Recommendation

Build the POC with:

| Decision | Choice |
|---|---|
| GUI | Gradio |
| Live voice model | `gemini-3.1-flash-live-preview` |
| Orchestration model | `gemini-3.1-flash-lite-preview` |
| Tool count | 2 |
| Connection pattern | Server-to-server from Python app server |
| Conversation model role | Voice interaction only |
| Flash-Lite role | Tool-routing and structured workflow control |
| State strategy | Backend session summary + minimal workflow state |
| Future hardening | Ephemeral tokens if moving toward direct browser-to-Live connections |

This is the cleanest POC design because it uses each Google model for what it is currently best at: Live for low-latency spoken interaction, and Flash-Lite for cheap, structured orchestration. ([Google AI for Developers][1])

---

[1]: https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-live-preview "Gemini 3.1 Flash Live Preview | Gemini API | Google AI for Developers"
[2]: https://ai.google.dev/gemini-api/docs/live-api "Gemini Live API overview | Gemini API | Google AI for Developers"
[3]: https://www.gradio.app/docs/gradio/audio "Gradio Docs"
[4]: https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input "st.audio_input - Streamlit Docs"
[5]: https://ai.google.dev/gemini-api/docs/live-api/tools "Tool use with Live API | Gemini API | Google AI for Developers"
[6]: https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-preview "Gemini 3.1 Flash-Lite Preview | Gemini API | Google AI for Developers"
[7]: https://ai.google.dev/gemini-api/docs/live-api/session-management "Session management with Live API | Gemini API | Google AI for Developers"
