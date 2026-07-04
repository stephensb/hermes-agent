# OBi200 → Asterisk → Hermes voice bridge

Lift the handset on an OBi200 (or any SIP ATA) and you're talking to Hermes.
This directory holds the Asterisk-side configuration; the Hermes-side adapter
lives in `../adapter.py`.

```
OBi200 (analog phone)                Asterisk PBX                 Hermes gateway
  off-hook auto-dials ──SIP──▶  registrar + Stasis dialplan
                                     │  ARI events + REST  ◀──────▶  SIPAdapter
                                     │  externalMedia (AudioSocket/TCP, 8 kHz SLIN)
                                     └────────── media ───────────▶  VAD→STT→agent→TTS
```

> **Just want to test it?** The `../e2e/` directory has a Docker Compose harness
> that runs Asterisk + Hermes + a simulated OBi200 (sipp) and places a real
> call end-to-end — no hardware needed. See `../e2e/README.md`.

## 1. Asterisk

Requires Asterisk 16+ with `res_ari`, `res_pjsip`, and `app_audiosocket`
(ships with modern Asterisk; `asterisk -rx "module show like audiosocket"`).

1. **ARI / HTTP** — enable the HTTP server in `/etc/asterisk/http.conf`:

   ```ini
   [general]
   enabled = yes
   bindaddr = 0.0.0.0
   bindport = 8088
   ```

   Then merge `ari.conf` here into `/etc/asterisk/ari.conf`. The `[hermes]`
   user's name/password become `SIP_ARI_USER` / `SIP_ARI_PASSWORD`.

2. **PJSIP endpoint** — merge `pjsip.conf` here into
   `/etc/asterisk/pjsip.conf`. Pick a real password for `obi200-auth`.

3. **Dialplan** — merge `extensions.conf` here into
   `/etc/asterisk/extensions.conf`. The `Stasis(hermes)` app name must equal
   `SIP_STASIS_APP` (default `hermes`).

4. Reload: `asterisk -rx "core reload"`.

## 2. OBi200

On stock firmware (no third-party flash needed):

1. **Service Provider (ITSP Profile / Voice Service → SP1)** — point it at
   Asterisk:
   - ProxyServer = Asterisk host/IP, ProxyServerPort = `5060`
   - AuthUserName = `obi200`, AuthPassword = the `obi200-auth` password
   - X_RegisterEnable = checked
2. **Auto-dial on off-hook** — under `Physical Interfaces → PHONE1 Port`, set
   `PrimaryLine = SP1`, and set the **DigitMap** / `OffHook AutoDial` so lifting
   the receiver immediately calls your Stasis target. The simplest is an
   off-hook hotline: set Port → `DigitMap` to a fixed string (e.g. `100`) and
   enable auto-dial, or set `OutboundCallRoute` to `{ph:100@SP1}`. `100` (or
   whatever you choose) just needs to match a dialplan extension — the
   `_X.` pattern in `extensions.conf` accepts any.

Lift the handset → the OBi200 registers/dials SP1 → Asterisk runs
`Stasis(hermes)` → the call connects to Hermes.

**Note:** the OBi200 endpoint (`pjsip.conf`) also lets Hermes call it back —
see "Outbound calls" below. No extra Asterisk config is needed for that; it
reuses the endpoint the phone already registered.

## 3. Hermes

Install the extra and set the environment (or run `hermes gateway setup` and
pick SIP):

```bash
pip install hermes-agent[sip]      # numpy; aiohttp + ffmpeg already present

export SIP_ARI_URL=http://ASTERISK_HOST:8088
export SIP_ARI_USER=hermes
export SIP_ARI_PASSWORD=...        # matches ari.conf
export SIP_STASIS_APP=hermes
export SIP_AUDIOSOCKET_ADVERTISE_HOST=HERMES_HOST   # reachable from Asterisk
export SIP_AUDIOSOCKET_PORT=9092
export SIP_ALLOWED_USERS=          # caller numbers; or SIP_ALLOW_ALL_USERS=true
# export SIP_BARGE_IN=true        # let the caller interrupt Hermes mid-reply

hermes gateway start
```

`SIP_AUDIOSOCKET_ADVERTISE_HOST` is the address **Asterisk** dials back to for
media, so it must be routable from the PBX to the Hermes host (use `127.0.0.1`
only when both run on the same machine). The bind host/port is where the
adapter listens.

## 4. Outbound calls (Hermes calls the OBi200)

Any tool call or cron job that sends a message to the SIP platform makes
Hermes **ring the phone** and speak the message once answered — the call
then continues as a normal live conversation until either side hangs up.

```python
# From a tool / cron job (same generic send_message_tool every platform uses):
send_message_tool(platform="sip", chat_id="obi200",
                  message="Reminder: your dentist appointment is in an hour.")
```

```bash
# Or via cron, with no explicit chat_id — rings SIP_HOME_CHANNEL
# (defaults to SIP_OUTBOUND_ENDPOINT, i.e. the OBi200):
cronjob(action="create", schedule="0 8 * * *", deliver="sip",
       prompt="Remind me to take my medication")
```

No Asterisk config changes are needed — outbound calls dial the same
`[obi200]` PJSIP endpoint via its existing registration (Asterisk reaches it
at the Contact address it registered from). Relevant env vars:

- `SIP_OUTBOUND_ENDPOINT` — which PJSIP endpoint to call (default: `obi200`).
- `SIP_OUTBOUND_TIMEOUT_SECONDS` — how long to ring before giving up (default: `30`).
- `SIP_HOME_CHANNEL` — the default destination for `deliver=sip` cron jobs
  (defaults to `SIP_OUTBOUND_ENDPOINT`).

If the phone doesn't answer (busy, no answer, rejected), the call is dropped
cleanly — no error is raised to the caller; check the gateway log for
`SIP: outbound call ... ended before answer`.

## 5. Real phone numbers (PSTN inbound)

To let people dial an actual phone number and reach Hermes — not just the
OBi200 — add a SIP trunk to a VoIP/ITSP provider. This is provider-specific
(registration vs. static-IP auth, DID formats all differ), so
`pjsip_trunk.conf.example` in this directory is a **template**, not a
drop-in config: fill in your provider's values, append the relevant section
to `pjsip.conf`, and add its `from-trunk` dialplan context to
`extensions.conf`. Once a trunk call reaches `Stasis(hermes)` it's identical
to an OBi200 call — same adapter, same pipeline, and `SIP_ALLOWED_USERS`
still gates who's allowed to talk to Hermes by caller ID.

## Notes & tuning

- **Codec**: the OBi200↔Asterisk leg uses G.711 (`ulaw`/`alaw`); Asterisk
  transcodes to 8 kHz signed-linear for AudioSocket. No codec config is needed
  on the Hermes side.
- **Turn-taking**: a caller's turn ends after `SIP_VAD_SILENCE_SECONDS` (1.5s
  default) of silence below `SIP_VAD_SILENCE_RMS` (200). Raise the RMS on a
  noisy line; lengthen the silence if callers get cut off mid-thought.
- **Half-duplex vs barge-in**: by default the caller's line is ignored while
  Hermes is speaking, so analog echo (ATA hybrid, speakerphone) can't make the
  bot interrupt itself. Set `SIP_BARGE_IN=true` to let the caller cut in
  mid-reply — only advisable on a clean handset with good echo cancellation.
- **Latency**: long agent turns (tool calls, memory recall) feel bad on a phone.
  The adapter's platform hint already asks for short spoken replies; for the
  snappiest experience pair SIP with a fast model and a lean toolset.
- STT/TTS use whatever providers the gateway is configured with
  (`tools/transcription_tools.py`, `tools/tts_tool.py`).
