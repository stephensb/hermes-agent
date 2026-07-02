# SIP bridge — end-to-end test harness

A Docker Compose stack that exercises the **whole** OBi200 → Asterisk → Hermes
path without any hardware:

```
caller (sipp)  ──SIP register + INVITE──▶  asterisk  ──ARI + AudioSocket──▶  hermes
   speaks a WAV as PCMU RTP  ─────────────▶  Stasis(hermes) → externalMedia → VAD→STT→agent→TTS
```

- **asterisk** — Debian Asterisk with this plugin's config templates
  (`../asterisk/*.conf`) mounted read-only.
- **hermes** — the gateway built from this repo, SIP plugin enabled, pointed at
  the Asterisk ARI and advertising its AudioSocket back to the `hermes` service.
- **caller** — `sipp` registers as the OBi200 (`obi200`), dials extension `100`
  (which the dialplan routes into `Stasis(hermes)`), and plays a phrase
  synthesized with `espeak-ng` and packed as 8 kHz PCMU RTP.

Everything runs on an internal bridge network, so no host ports are needed
except ARI on `127.0.0.1:8088` for debugging.

## Run it

From this directory:

```bash
# 1. (Optional) give Hermes a model so it can actually reply.
cp hermes.env.example hermes.env && $EDITOR hermes.env

# 2. Bring up the PBX and the gateway.
docker compose up --build -d asterisk hermes

# 3. Wait for the bridge to connect.
docker compose logs -f hermes      # look for: "SIP: connected to ARI ... AudioSocket on 0.0.0.0:9092"

# 4. Place one test call.
docker compose run --rm caller

# 5. Watch what happened.
docker compose logs hermes         # StasisStart, media attached, transcript, reply
```

Tear down with `docker compose down -v`.

## What each stage proves

| You see in `hermes` logs | Validates |
|--------------------------|-----------|
| `SIP: connected to ARI ... AudioSocket on ...` | ARI websocket + AudioSocket server came up |
| `SIP: incoming call ... from <redacted>` | REGISTER + INVITE reached Stasis; caller admitted |
| `SIP: media attached for call ...` | externalMedia channel dialed the AudioSocket server and the UUID handshake matched — **the core integration** |
| `SIP: call ... caller said: hello hermes ...` | RTP → SLIN → VAD turn → STT round-trip works |
| a spoken reply on the (silent) call + `call ... ended` | agent turn → TTS → AudioSocket playback → clean teardown |

The first three lines validate the plumbing **even without a model key** — they
don't depend on the LLM. Lines 4–5 need STT and a provider configured in
`hermes.env` (see the comments there); without them the call still connects and
you'll see the transcription/agent step fail rather than a reply.

## Notes & troubleshooting

- **No model key?** The harness still verifies register → Stasis → externalMedia
  → AudioSocket. The agent turn simply errors at the LLM call; that's expected.
- **Auth 407 vs 401.** Some Asterisk builds challenge with `407` instead of
  `401`. If `caller` fails auth, change the two `response="401"` lines in
  `caller/uac.xml` to `407`.
- **Change what's said / dialed.** Override on the caller:
  `docker compose run --rm -e SAY="what's the weather" -e DIAL_EXT=100 caller`.
- **AudioSocket module missing.** If Asterisk logs `No such command 'AudioSocket'`,
  the image's Asterisk lacks `app_audiosocket`/`chan_audiosocket`; rebuild
  `asterisk/Dockerfile` from a build that includes them (Debian bookworm's
  package does).
- **Use a real softphone instead of sipp.** Point Linphone/Zoiper at
  `asterisk` (user `obi200` / `change-me`), register, and dial `100` — same
  path, live two-way audio.
- **Prod is not this.** `SIP_ALLOW_ALL_USERS=true` and the `change-me`
  passwords are for the test only. See `../asterisk/README.md` for the real
  OBi200 + lockdown setup.
