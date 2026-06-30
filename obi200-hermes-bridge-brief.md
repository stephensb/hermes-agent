# OBi200 → Hermes Agent voice bridge — handoff brief

## Goal
Wire a Polycom/Poly OBi200 (analog phone via SIP ATA) so lifting the receiver
starts a live conversation with Hermes Agent — same idea as the FeTAp-phone-
to-Claude project, but the OBi200 already speaks SIP/RTP natively, so no
ESP32/audio hardware hacking is needed.

## Architecture decided so far
- OBi200 stays on stock firmware. Configure a custom ITSP/SIP profile +
  auto-dial digitmap so off-hook immediately places a call to our SIP server.
  (Third-party OBi firmware is only needed for SSH/zero-touch bypass — not
  required for this.)
- Asterisk (or FreeSWITCH) as the SIP/RTP registrar. OBi200 registers/dials
  into it. Asterisk's ARI `externalMedia` streams call audio over a
  WebSocket to a new bridge service.
- That bridge is a **new platform adapter** in hermes-agent, same pattern as
  the existing Discord/Telegram gateway adapters — not a fork of the CLI's
  `voice_mode.py` script.

## What to reuse from hermes-agent
- `tools/transcription_tools.py` — STT provider abstraction (local
  faster-whisper / Groq / OpenAI / Mistral / xAI), format validation.
- TTS sentence-buffering/streaming logic — speaks as text streams in rather
  than waiting for the full response.
- The shared `AIAgent` conversation loop (frontend-agnostic — CLI, Discord,
  Telegram all call into the same core).
- `gateway/platforms/discord.py`'s `VoiceReceiver` as the closest existing
  analog: continuous full-duplex network audio, per-session handling,
  decode → STT → agent → TTS → speak-back. Closer in shape to telephony
  than the CLI's push-to-talk mic loop.

## What to port (not reuse directly)
- The continuous-mode VAD/silence-detection algorithm from `voice_mode.py`
  (RMS threshold ~200, 0.3s speech-confirm, 3s silence-to-end). A handset
  has no push-to-talk key, so this drives turn-taking — needs retuning for
  phone-line noise floor vs. room-mic noise floor.

## What's genuinely new
- The Asterisk-side adapter: buffer incoming SLIN16 PCM from `externalMedia`
  into WAV chunks per utterance for `transcription_tools`. On the way back,
  resample/frame TTS output to match Asterisk's expected PCM format and
  20ms RTP packetization.

## Watch-outs
- Full `AIAgent` loop latency (tool calls, memory recall) can blow past the
  ~2s threshold where phone conversations start feeling bad. Consider a
  stripped "phone mode" profile: fewer tools, short system prompt enforcing
  brief replies.
- Use a **soft prompt instruction** for brevity, not a hard `max_tokens`
  cutoff — a hard cutoff mid-sentence breaks turn-taking state cleanly
  (this bit the original FeTAp-phone project the same way).

## Open question for the fork
Confirm actual module paths/class names in the forked repo match the above
(docs/DeepWiki were the source, not the source tree itself) before writing
the adapter.
