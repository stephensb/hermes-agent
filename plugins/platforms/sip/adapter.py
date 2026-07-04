"""
SIP voice-bridge platform adapter for Hermes Agent.

Lets an analog phone behind a SIP ATA (e.g. a Poly/OBi200) hold a live spoken
conversation with Hermes.  The phone registers and dials into an Asterisk PBX;
Asterisk runs a Stasis dialplan that hands the call to this adapter over two
channels:

  * **ARI** (Asterisk REST Interface) — call control.  We subscribe to the
    Stasis application's event WebSocket and drive answer / bridge / hangup
    via the REST API.
  * **AudioSocket** (TCP) — raw media.  Asterisk's ``externalMedia`` channel
    connects to a TCP server we run here and streams 8 kHz signed-linear
    16-bit mono PCM in 20 ms (320-byte) frames.  We stream synthesized speech
    back over the same socket.

Audio flow (inbound — caller dials in)::

    caller speaks ─▶ AudioSocket PCM ─▶ VAD turn detector ─▶ WAV
                  ─▶ transcribe_audio() ─▶ handle_message() (agent loop)
    agent reply  ─▶ send() ─▶ say queue ─▶ TTS ─▶ ffmpeg 8k s16le
                  ─▶ PCM queue ─▶ 320-byte frames ─▶ AudioSocket

Replies are spoken through a two-stage per-call pipeline (a synthesis task
feeding a playout task through bounded queues) so that multiple ``send()``
calls in one agent turn are spoken **in order** instead of cancelling each
other, and sentence N+1 synthesizes while sentence N is still playing.

Outbound calls (Hermes rings the phone) work the other direction: any tool or
cron job that calls ``send_message_tool`` / ``cronjob(deliver="sip", ...)``
ends up at ``SIPAdapter.send(chat_id, content)`` like every other platform.
If ``chat_id`` isn't a live call, ``send()`` treats it as a PJSIP endpoint
name and originates a call via ARI (``originate_call``); once the phone
answers, the *same* AudioSocket/VAD/TTS pipeline as an inbound call takes
over, speaking ``content`` first and then continuing as a live conversation.

The adapter is a plugin: it subclasses ``BasePlatformAdapter`` and registers
via ``register(ctx)`` with zero changes to core Hermes code.  See
``asterisk/README.md`` in this directory for the PBX + OBi200 wiring, and
``asterisk/pjsip_trunk.conf.example`` for routing real PSTN numbers in.

Configuration via environment variables (or ``config.yaml`` ``extra:``)::

    SIP_ARI_URL                 http://127.0.0.1:8088   (Asterisk ARI base)
    SIP_ARI_USER                ARI username (ari.conf)
    SIP_ARI_PASSWORD            ARI password (ari.conf)
    SIP_STASIS_APP              Stasis app name        (default: hermes)
    SIP_AUDIOSOCKET_HOST        TCP bind host          (default: 0.0.0.0)
    SIP_AUDIOSOCKET_PORT        TCP bind port          (default: 9092)
    SIP_AUDIOSOCKET_ADVERTISE_HOST  address Asterisk dials back to
                                (default: 127.0.0.1)
    SIP_ALLOWED_USERS           comma-separated caller numbers allowed
    SIP_ALLOW_ALL_USERS         allow any caller (dev only)
    SIP_BARGE_IN                let caller speech interrupt playback
                                (default: false — half-duplex; the caller's
                                line is ignored while Hermes is speaking, so
                                analog echo can't trigger self-interruption)
    SIP_VAD_SILENCE_RMS         end-of-turn RMS threshold (default: 200)
    SIP_VAD_SILENCE_SECONDS     silence to end a turn      (default: 1.5)
    SIP_OUTBOUND_ENDPOINT       PJSIP endpoint Hermes calls out to
                                (default: obi200)
    SIP_OUTBOUND_TIMEOUT_SECONDS  ring timeout for outbound calls
                                (default: 30)
    SIP_HOME_CHANNEL            default cron/send_message destination
                                (default: SIP_OUTBOUND_ENDPOINT)
"""

import asyncio
import json
import logging
import os
import struct
import tempfile
import time
import uuid as uuid_mod
import wave
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.platforms.helpers import redact_phone
from gateway.config import Platform

logger = logging.getLogger(__name__)

# numpy is optional (the [sip] extra provides it).  Import once at module
# load — pcm_rms runs on every 20 ms frame, and a failed ``import numpy``
# inside the function would re-run the import machinery 50×/sec per call.
try:
    import numpy as _np
except Exception:  # pragma: no cover - environment-dependent
    _np = None

# Project-wide speech/silence RMS boundary.  tools/voice_mode.py owns the
# constant; fall back to its documented value if the voice module moves.
try:
    from tools.voice_mode import SILENCE_RMS_THRESHOLD as _DEFAULT_SILENCE_RMS
except Exception:  # pragma: no cover - environment-dependent
    _DEFAULT_SILENCE_RMS = 200


# ---------------------------------------------------------------------------
# AudioSocket framing  (https://docs.asterisk.org/ AudioSocket protocol)
#
# Wire format per frame:  <1 byte kind><2 byte big-endian length><payload>
# ---------------------------------------------------------------------------

AS_KIND_HANGUP = 0x00   # Asterisk → us: the call has ended
AS_KIND_UUID = 0x01     # Asterisk → us (first frame): 16-byte call UUID
AS_KIND_SILENCE = 0x02  # informational silence marker
AS_KIND_SLIN = 0x10     # audio payload, signed-linear 16-bit 8 kHz mono
AS_KIND_ERROR = 0xff    # Asterisk → us: error

# 8 kHz * 20 ms * 2 bytes/sample = 320 bytes per outbound frame.
SLIN_SAMPLE_RATE = 8000
SLIN_FRAME_MS = 20
SLIN_FRAME_BYTES = SLIN_SAMPLE_RATE * SLIN_FRAME_MS // 1000 * 2  # 320

# Constant header for full outbound audio frames (precomputed once — the
# playout loop writes 50 frames/sec per call).
_SLIN_FRAME_HEADER = struct.pack(">BH", AS_KIND_SLIN, SLIN_FRAME_BYTES)


def parse_audiosocket_frames(buffer: bytes) -> Tuple[List[Tuple[int, bytes]], bytes]:
    """Split a TCP byte buffer into complete AudioSocket frames.

    Returns ``(frames, remainder)`` where *frames* is a list of
    ``(kind, payload)`` tuples for every complete frame found and *remainder*
    is the trailing bytes of an incomplete frame (to be prepended to the next
    read).  Pure function — no I/O — so it is unit-testable against partial
    reads.
    """
    frames: List[Tuple[int, bytes]] = []
    offset = 0
    n = len(buffer)
    while n - offset >= 3:
        kind = buffer[offset]
        length = (buffer[offset + 1] << 8) | buffer[offset + 2]
        if n - offset - 3 < length:
            break  # payload not fully arrived yet
        payload = buffer[offset + 3 : offset + 3 + length]
        frames.append((kind, payload))
        offset += 3 + length
    return frames, buffer[offset:]


def encode_audiosocket_frame(kind: int, payload: bytes = b"") -> bytes:
    """Encode a single AudioSocket frame."""
    if len(payload) > 0xFFFF:
        raise ValueError("AudioSocket payload exceeds 65535 bytes")
    return struct.pack(">BH", kind, len(payload)) + payload


def frame_pcm(pcm: bytes, frame_bytes: int = SLIN_FRAME_BYTES) -> List[bytes]:
    """Slice raw PCM into fixed-size frames, zero-padding the final frame.

    Pure helper used when packetizing synthesized speech for AudioSocket.
    """
    if frame_bytes <= 0:
        raise ValueError("frame_bytes must be positive")
    out: List[bytes] = []
    view = memoryview(pcm)
    for i in range(0, len(pcm), frame_bytes):
        chunk = bytes(view[i : i + frame_bytes])
        if len(chunk) < frame_bytes:
            chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
        out.append(chunk)
    return out


def pcm_rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of signed-linear 16-bit mono PCM.

    Uses numpy when available (fast path); otherwise a zero-copy
    memoryview cast.  Runs on every 20 ms inbound frame, so no imports or
    large temporaries here.
    """
    if not pcm:
        return 0.0
    count = len(pcm) // 2
    if count == 0:
        return 0.0
    if _np is not None:
        try:
            samples = _np.frombuffer(pcm, dtype=_np.int16).astype(_np.float64)
            return float(_np.sqrt(_np.mean(samples * samples)))
        except Exception:
            pass  # fall through to the pure-Python path
    total = 0
    for s in memoryview(pcm)[: count * 2].cast("h"):
        total += s * s
    return (total / count) ** 0.5


# ---------------------------------------------------------------------------
# Voice-activity turn detector
#
# Ported from tools/voice_mode.py's continuous-mode silence detection and
# retuned for a phone line (no push-to-talk key drives turn-taking).  Drop the
# default silence-to-end from voice_mode's 3.0 s to 1.5 s — phone conversation
# pacing feels laggy above ~2 s — but keep it configurable.
# ---------------------------------------------------------------------------

class PhoneTurnDetector:
    """Accumulate caller PCM and emit one buffered utterance per turn.

    Feed fixed-cadence audio frames via :meth:`feed`; it returns the buffered
    utterance PCM (bytes) when the caller has spoken and then gone silent for
    ``silence_seconds``, otherwise ``None``.  A burst shorter than
    ``min_speech_seconds`` is treated as noise and discarded.

    Buffering starts at the FIRST speech frame (tentatively) and a short
    pre-roll of the preceding silence is prepended, so the leading syllable
    of an utterance is never clipped while speech is still being confirmed.
    ``max_utterance_seconds`` bounds the buffer: continuous sound above the
    threshold (hold music, line hum) force-emits rather than growing forever.
    """

    def __init__(
        self,
        *,
        sample_rate: int = SLIN_SAMPLE_RATE,
        silence_rms: float = float(_DEFAULT_SILENCE_RMS),
        silence_seconds: float = 1.5,
        min_speech_seconds: float = 0.3,
        preroll_seconds: float = 0.2,
        max_utterance_seconds: float = 60.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_rms = silence_rms
        self.silence_seconds = silence_seconds
        self.min_speech_seconds = min_speech_seconds
        self.preroll_seconds = preroll_seconds
        self.max_utterance_seconds = max_utterance_seconds
        # Pre-roll ring buffer sized in 20 ms frames.
        frames = max(1, int(preroll_seconds * 1000 / SLIN_FRAME_MS))
        self._preroll: deque = deque(maxlen=frames)
        self.reset()

    def reset(self) -> None:
        self._buf = bytearray()
        self._buf_seconds = 0.0
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0
        self._in_speech = False
        # NOTE: the pre-roll deque survives reset on purpose — it holds the
        # most recent line audio regardless of turn boundaries.

    def _frame_seconds(self, pcm: bytes) -> float:
        samples = len(pcm) // 2
        return samples / float(self.sample_rate) if self.sample_rate else 0.0

    def _emit(self) -> bytes:
        utterance = bytes(self._buf)
        self.reset()
        return utterance

    def feed(self, pcm: bytes) -> Optional[bytes]:
        """Process one audio frame.  Returns a finished utterance or None."""
        if not pcm:
            return None
        dur = self._frame_seconds(pcm)
        rms = pcm_rms(pcm)
        is_speech = rms >= self.silence_rms

        if is_speech:
            if not self._buf and not self._in_speech:
                # First (tentative) speech frame: prepend the pre-roll so the
                # onset isn't clipped even before speech is confirmed.
                for prev in self._preroll:
                    self._buf.extend(prev)
                    self._buf_seconds += self._frame_seconds(prev)
                self._preroll.clear()
            self._speech_seconds += dur
            self._silence_seconds = 0.0
            if self._speech_seconds >= self.min_speech_seconds:
                self._in_speech = True
            # Buffer from the first speech frame — discarded later if the
            # burst never confirms (see the blip branch below).
            self._buf.extend(pcm)
            self._buf_seconds += dur
            if self._buf_seconds >= self.max_utterance_seconds and self._in_speech:
                return self._emit()
            return None

        # Silence frame.
        if self._in_speech:
            self._buf.extend(pcm)  # keep trailing silence inside the utterance
            self._buf_seconds += dur
            self._silence_seconds += dur
            if self._silence_seconds >= self.silence_seconds:
                return self._emit()
        else:
            # A sub-min_speech blip followed by silence: noise.  Drop the
            # tentative buffer and keep rolling pre-roll context instead.
            self._speech_seconds = 0.0
            if self._buf:
                self._buf = bytearray()
                self._buf_seconds = 0.0
            self._preroll.append(pcm)
        return None


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = SLIN_SAMPLE_RATE) -> bytes:
    """Wrap mono 16-bit PCM in a WAV container (for the STT file API)."""
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _env_or_extra(extra: dict, env: str, key: str, default: Any = "") -> Any:
    val = os.getenv(env)
    if val is not None and val != "":
        return val
    return extra.get(key, default)


def _ari_configured(extra: dict) -> bool:
    """True when the three required ARI settings are present (env or extra)."""
    return bool(
        _env_or_extra(extra, "SIP_ARI_URL", "ari_url")
        and _env_or_extra(extra, "SIP_ARI_USER", "ari_user")
        and _env_or_extra(extra, "SIP_ARI_PASSWORD", "ari_password")
    )


# ---------------------------------------------------------------------------
# Per-call state
# ---------------------------------------------------------------------------

class _Call:
    """Bookkeeping for one in-flight phone call."""

    def __init__(self, channel_id: str, caller: str,
                 detector: Optional[PhoneTurnDetector] = None,
                 *, outbound: bool = False) -> None:
        self.channel_id = channel_id
        self.caller = caller or "unknown"
        # Canonical UUID string (8-4-4-4-12).  Asterisk's externalMedia ``data``
        # field expects a valid UUID and the AudioSocket ID frame echoes it back
        # as 16 raw bytes, which we re-render to this same canonical form.
        self.media_uuid = str(uuid_mod.uuid4())
        self.bridge_id: Optional[str] = None
        self.external_channel_id: Optional[str] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.detector = detector or PhoneTurnDetector()
        # Two-stage playback pipeline (see _synth_loop / _playout_loop).
        self.say_queue: asyncio.Queue = asyncio.Queue()
        self.pcm_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        self.synth_task: Optional[asyncio.Task] = None
        self.playout_task: Optional[asyncio.Task] = None
        self.playing = False           # a PCM chunk is currently streaming
        self.flush_generation = 0      # bumped by barge-in to abort playout
        # True for calls Hermes originated (rings the OBi200) rather than
        # calls the OBi200 placed to Hermes.  Outbound calls skip the
        # caller-admission check (we placed the call) and are already
        # answered by the time they enter Stasis.
        self.outbound = outbound
        self.utterance_tasks: Set[asyncio.Task] = set()

    def is_speaking(self) -> bool:
        """True while any queued or in-flight reply audio remains."""
        return (self.playing
                or not self.pcm_queue.empty()
                or not self.say_queue.empty())


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class SIPAdapter(BasePlatformAdapter):
    """SIP voice bridge adapter (Asterisk ARI + AudioSocket)."""

    # A phone call has no message editing; opting out keeps the gateway's
    # streaming consumer from delivering partial chunks as separate send()
    # calls (which would be spoken as stuttering, repeated speech).
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("sip"))
        extra = getattr(config, "extra", {}) or {}

        self.ari_url = str(_env_or_extra(extra, "SIP_ARI_URL", "ari_url",
                                         "http://127.0.0.1:8088")).rstrip("/")
        self.ari_user = str(_env_or_extra(extra, "SIP_ARI_USER", "ari_user", ""))
        self.ari_password = str(_env_or_extra(extra, "SIP_ARI_PASSWORD", "ari_password", ""))
        self.stasis_app = str(_env_or_extra(extra, "SIP_STASIS_APP", "stasis_app", "hermes"))

        self.bind_host = str(_env_or_extra(extra, "SIP_AUDIOSOCKET_HOST",
                                           "audiosocket_host", "0.0.0.0"))
        try:
            self.bind_port = int(_env_or_extra(extra, "SIP_AUDIOSOCKET_PORT",
                                               "audiosocket_port", 9092))
        except (TypeError, ValueError):
            self.bind_port = 9092
        self.advertise_host = str(_env_or_extra(
            extra, "SIP_AUDIOSOCKET_ADVERTISE_HOST", "audiosocket_advertise_host",
            "127.0.0.1"))

        # VAD tunables
        try:
            self.vad_silence_rms = float(_env_or_extra(
                extra, "SIP_VAD_SILENCE_RMS", "vad_silence_rms",
                float(_DEFAULT_SILENCE_RMS)))
        except (TypeError, ValueError):
            self.vad_silence_rms = float(_DEFAULT_SILENCE_RMS)
        try:
            self.vad_silence_seconds = float(_env_or_extra(
                extra, "SIP_VAD_SILENCE_SECONDS", "vad_silence_seconds", 1.5))
        except (TypeError, ValueError):
            self.vad_silence_seconds = 1.5

        # Half-duplex by default: while Hermes is speaking, ignore the
        # caller's line so analog echo (ATA hybrid, speakerphone) can't be
        # mistaken for a barge-in and trigger self-interruption loops.
        from utils import is_truthy_value
        self.barge_in = is_truthy_value(
            _env_or_extra(extra, "SIP_BARGE_IN", "barge_in", ""))

        # Call-admission allowlist (checked at StasisStart, BEFORE the call
        # is answered — the gateway authz layer still gates every message as
        # defense in depth, but admission control means unauthorized callers
        # never consume STT spend or hold a bridge open).
        allowed_raw = _env_or_extra(extra, "SIP_ALLOWED_USERS", "allowed_users", "")
        if isinstance(allowed_raw, str):
            self.allowed_callers = {c.strip() for c in allowed_raw.split(",") if c.strip()}
        else:
            self.allowed_callers = {str(c).strip() for c in (allowed_raw or []) if str(c).strip()}
        self.allow_all_callers = is_truthy_value(
            _env_or_extra(extra, "SIP_ALLOW_ALL_USERS", "allow_all_users", ""))

        # Outbound calling (Hermes rings the phone) — see originate_call().
        self.outbound_endpoint = str(_env_or_extra(
            extra, "SIP_OUTBOUND_ENDPOINT", "outbound_endpoint", "obi200"))
        try:
            self.outbound_timeout_seconds = int(_env_or_extra(
                extra, "SIP_OUTBOUND_TIMEOUT_SECONDS", "outbound_timeout_seconds", 30))
        except (TypeError, ValueError):
            self.outbound_timeout_seconds = 30

        # Runtime state
        self._session = None  # aiohttp.ClientSession
        self._ws = None       # aiohttp ARI event websocket
        self._ws_task: Optional[asyncio.Task] = None
        self._server: Optional[asyncio.AbstractServer] = None
        self._calls: Dict[str, _Call] = {}            # channel_id -> _Call
        self._uuid_to_channel: Dict[str, str] = {}    # media_uuid -> channel_id
        self._external_channels: set = set()          # externalMedia channel ids
        self._external_owner: Dict[str, str] = {}     # ext channel id -> caller channel id
        # channel_id -> message to speak, for calls Hermes originated that
        # haven't reached StasisStart yet (ringing/dialing).  Populated before
        # the ARI create-channel POST so a StasisStart racing the POST
        # response is still recognized as ours (same defensive pattern as
        # _external_channels for the externalMedia leg).
        self._pending_outbound: Dict[str, Optional[str]] = {}

    @property
    def name(self) -> str:
        return "SIP"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not (self.ari_url and self.ari_user and self.ari_password):
            logger.error("SIP: SIP_ARI_URL, SIP_ARI_USER and SIP_ARI_PASSWORD must be set")
            self._set_fatal_error("config_missing",
                                  "ARI url/user/password not configured",
                                  retryable=False)
            return False
        try:
            import aiohttp
        except ImportError:
            logger.error("SIP: aiohttp is required (pip install hermes-agent[sip])")
            self._set_fatal_error("missing_dep", "aiohttp not installed", retryable=False)
            return False

        # Start the AudioSocket TCP media server first so Asterisk can dial
        # back the moment we create an externalMedia channel.
        try:
            self._server = await asyncio.start_server(
                self._handle_audiosocket_conn, self.bind_host, self.bind_port)
        except OSError as e:
            logger.error("SIP: cannot bind AudioSocket %s:%s — %s",
                         self.bind_host, self.bind_port, e)
            self._set_fatal_error("bind_failed", str(e), retryable=True)
            return False

        # Basic auth on the session covers both the REST calls and the
        # events-websocket handshake — never put the password in the URL,
        # where it would leak into logged exception messages.
        self._session = aiohttp.ClientSession(
            auth=aiohttp.BasicAuth(self.ari_user, self.ari_password))

        # Open the ARI event websocket subscribed to our Stasis app only
        # (no subscribeAll: PBX-wide events would burn event-loop time that
        # the 20 ms media path needs).
        ws_base = self.ari_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_base}/ari/events?app={self.stasis_app}"
        try:
            self._ws = await self._session.ws_connect(ws_url, heartbeat=30)
        except Exception as e:
            logger.error("SIP: ARI websocket connect failed — %s", e)
            await self._teardown_transports()
            self._set_fatal_error("ari_connect_failed", str(e), retryable=True)
            return False

        self._ws_task = asyncio.create_task(self._ari_event_loop())
        self._mark_connected()
        logger.info("SIP: connected to ARI %s (app=%s), AudioSocket on %s:%s",
                    self.ari_url, self.stasis_app, self.bind_host, self.bind_port)
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        # Hang up any live calls.
        for channel_id in list(self._calls):
            await self._end_call(channel_id, hangup=True)
        await self._teardown_transports()

    async def _teardown_transports(self) -> None:
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        self._ws_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    # ── ARI: call control ─────────────────────────────────────────────────

    async def _ari_request(self, method: str, path: str,
                           params: Optional[dict] = None,
                           quiet_404: bool = False) -> Any:
        """Issue an ARI REST request; returns parsed JSON or None.

        ``quiet_404`` suppresses the warning for expected-missing resources
        (idempotent teardown DELETEs racing Asterisk's own cleanup).
        """
        if self._session is None:
            return None
        url = f"{self.ari_url}/ari/{path.lstrip('/')}"
        try:
            async with self._session.request(method, url, params=params) as resp:
                if resp.status >= 400:
                    if not (quiet_404 and resp.status == 404):
                        body = await resp.text()
                        logger.warning("SIP: ARI %s %s → %s: %s",
                                       method, path, resp.status, body[:200])
                    return None
                if resp.content_type == "application/json":
                    return await resp.json()
                return await resp.text()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("SIP: ARI %s %s failed — %s", method, path, e)
            return None

    async def _ari_event_loop(self) -> None:
        import aiohttp
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        event = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    try:
                        await self._dispatch_ari_event(event)
                    except Exception:
                        logger.exception("SIP: error handling ARI event")
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("SIP: ARI event loop error — %s", e)
        finally:
            if self.is_connected:
                self._set_fatal_error("ari_lost", "ARI websocket closed", retryable=True)
                await self._notify_fatal_error()

    async def _dispatch_ari_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "StasisStart":
            await self._on_stasis_start(event)
        elif etype in ("StasisEnd", "ChannelDestroyed", "ChannelHangupRequest"):
            channel = event.get("channel", {})
            cid = channel.get("id")
            if cid in self._calls:
                await self._end_call(cid, hangup=False)
            elif cid in self._external_owner:
                # The media leg died first (TCP reset, Asterisk error): the
                # caller would otherwise sit on a silent line forever.  Hang
                # the whole call up.
                owner = self._external_owner.get(cid)
                if owner in self._calls:
                    logger.warning("SIP: media leg for call %s died — ending call", owner)
                    await self._end_call(owner, hangup=True)
            elif cid in self._pending_outbound:
                # An originated call that never reached StasisStart — busy,
                # no answer, or rejected.  Asterisk routes these events to us
                # because the channel was created with app=<our app>, even
                # though it never actually entered the app.
                logger.info("SIP: outbound call %s ended before answer (%s)", cid, etype)
                self._pending_outbound.pop(cid, None)
            self._external_channels.discard(cid)
            self._external_owner.pop(cid, None)

    def _caller_allowed(self, caller: str) -> bool:
        """Call-admission check, mirroring the gateway env-allowlist policy."""
        if self.allow_all_callers:
            return True
        if os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}:
            return True
        return bool(caller) and caller in self.allowed_callers

    async def _on_stasis_start(self, event: dict) -> None:
        channel = event.get("channel", {})
        channel_id = channel.get("id")
        if not channel_id:
            return
        # Ignore our own externalMedia leg re-entering Stasis.  The id is
        # pre-assigned before the POST (see below) so this check does not
        # race the POST response; the name-prefix test is a fallback for
        # channels created outside this adapter.
        if channel_id in self._external_channels:
            return
        name = channel.get("name", "")
        if name.startswith("AudioSocket") or name.startswith("UnicastRTP"):
            self._external_channels.add(channel_id)
            return

        if channel_id in self._pending_outbound:
            # A call Hermes originated (see originate_call) just answered.
            # It's already Up (the far end sent 200 OK) — no explicit answer
            # needed, and it's inherently authorized: we placed it.
            message = self._pending_outbound.pop(channel_id)
            call = _Call(channel_id, self.outbound_endpoint, outbound=True,
                        detector=PhoneTurnDetector(
                            silence_rms=self.vad_silence_rms,
                            silence_seconds=self.vad_silence_seconds))
            self._calls[channel_id] = call
            self._uuid_to_channel[call.media_uuid] = channel_id
            if message:
                # Enqueue now (not at media-attach time) so it stays first in
                # line even if another send() races in before AudioSocket
                # connects — say_queue is a plain FIFO, order doesn't depend
                # on when the synth/playout tasks start draining it.
                call.say_queue.put_nowait(message)
            logger.info("SIP: outbound call %s to %s answered", channel_id,
                       redact_phone(self.outbound_endpoint))
            await self._attach_external_media_and_bridge(call)
            return

        caller = (channel.get("caller") or {}).get("number") or ""
        if not self._caller_allowed(caller):
            # Reject BEFORE answering: unauthorized callers never consume
            # STT spend, never hold a bridge, and never reach the gateway's
            # pairing prompt (which would read a pairing code aloud to a
            # stranger).
            logger.warning("SIP: rejecting unauthorized caller %s",
                           redact_phone(caller))
            await self._ari_request("DELETE", f"channels/{channel_id}",
                                    quiet_404=True)
            return

        call = _Call(channel_id, caller, detector=PhoneTurnDetector(
            silence_rms=self.vad_silence_rms,
            silence_seconds=self.vad_silence_seconds))
        self._calls[channel_id] = call
        self._uuid_to_channel[call.media_uuid] = channel_id

        logger.info("SIP: incoming call %s from %s",
                    channel_id, redact_phone(call.caller))

        # Answer, then bridge the caller with an externalMedia (AudioSocket) leg.
        await self._ari_request("POST", f"channels/{channel_id}/answer")
        if channel_id not in self._calls:
            # Caller hung up while we were answering — nothing to clean up
            # beyond what _end_call already did.
            return
        await self._attach_external_media_and_bridge(call)

    async def _attach_external_media_and_bridge(self, call: _Call) -> None:
        """Create the externalMedia (AudioSocket) leg and bridge it to *call*.

        Shared by the inbound (caller dialed in) and outbound (Hermes
        originated the call) paths — both need identical media plumbing once
        the caller-facing channel is answered and admitted.
        """
        channel_id = call.channel_id

        # Pre-assign the externalMedia channel id and register it BEFORE the
        # POST: its StasisStart can arrive before the POST response, and the
        # id (not a fragile name prefix) is what keeps it from being
        # misclassified as a new inbound call.
        ext_id = f"sip-media-{call.media_uuid}"
        call.external_channel_id = ext_id
        self._external_channels.add(ext_id)
        self._external_owner[ext_id] = channel_id

        ext = await self._ari_request("POST", "channels/externalMedia", params={
            "channelId": ext_id,
            "app": self.stasis_app,
            "external_host": f"{self.advertise_host}:{self.bind_port}",
            "format": "slin",
            "encapsulation": "audiosocket",
            "transport": "tcp",
            "connection_type": "client",
            "data": call.media_uuid,
        })
        if not ext or "id" not in ext:
            logger.error("SIP: externalMedia creation failed for %s", channel_id)
            await self._end_call(channel_id, hangup=True)
            return
        if channel_id not in self._calls:
            # Caller hung up mid-setup: _end_call already ran and won't have
            # seen the media leg attach, so drop it explicitly.
            await self._ari_request("DELETE", f"channels/{ext_id}", quiet_404=True)
            return

        bridge = await self._ari_request("POST", "bridges", params={"type": "mixing"})
        if not bridge or "id" not in bridge:
            logger.error("SIP: bridge creation failed for %s", channel_id)
            await self._end_call(channel_id, hangup=True)
            return
        call.bridge_id = bridge["id"]
        if channel_id not in self._calls:
            await self._ari_request("DELETE", f"bridges/{bridge['id']}", quiet_404=True)
            await self._ari_request("DELETE", f"channels/{ext_id}", quiet_404=True)
            return
        await self._ari_request(
            "POST", f"bridges/{call.bridge_id}/addChannel",
            params={"channel": f"{channel_id},{ext_id}"})

    async def _end_call(self, channel_id: str, *, hangup: bool) -> None:
        call = self._calls.pop(channel_id, None)
        if call is None:
            return
        self._uuid_to_channel.pop(call.media_uuid, None)
        for task in (call.synth_task, call.playout_task, *call.utterance_tasks):
            if task and not task.done():
                task.cancel()
        call.utterance_tasks.clear()
        if call.writer is not None:
            try:
                call.writer.close()
            except Exception:
                pass
        if call.bridge_id:
            await self._ari_request("DELETE", f"bridges/{call.bridge_id}",
                                    quiet_404=True)
        if call.external_channel_id:
            # Normally the TCP close above hangs the media leg up; the
            # explicit DELETE covers the setup paths where the socket never
            # attached (quiet_404 keeps the common already-gone case silent).
            await self._ari_request(
                "DELETE", f"channels/{call.external_channel_id}", quiet_404=True)
            self._external_channels.discard(call.external_channel_id)
            self._external_owner.pop(call.external_channel_id, None)
        if hangup:
            await self._ari_request("DELETE", f"channels/{channel_id}",
                                    quiet_404=True)
        logger.info("SIP: call %s ended", channel_id)

    # ── AudioSocket: media ────────────────────────────────────────────────

    async def _handle_audiosocket_conn(self, reader: asyncio.StreamReader,
                                       writer: asyncio.StreamWriter) -> None:
        """Handle one Asterisk externalMedia TCP connection."""
        call: Optional[_Call] = None
        buf = b""
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                buf += data
                frames, buf = parse_audiosocket_frames(buf)
                for kind, payload in frames:
                    if kind == AS_KIND_UUID:
                        call = self._attach_media(payload, writer)
                    elif kind == AS_KIND_SLIN and call is not None:
                        self._on_caller_audio(call, payload)
                    elif kind == AS_KIND_HANGUP:
                        if call is not None:
                            await self._end_call(call.channel_id, hangup=True)
                            call = None
                        return
                    elif kind == AS_KIND_ERROR:
                        logger.warning("SIP: AudioSocket error frame: %r", payload[:64])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("SIP: AudioSocket connection error — %s", e)
        finally:
            try:
                writer.close()
            except Exception:
                pass
            # Media socket gone: if the call is still live, the caller is on
            # a dead line — hang the whole call up rather than leaving a
            # zombie with a closed writer.
            if call is not None and call.channel_id in self._calls:
                logger.warning("SIP: media socket for call %s closed — ending call",
                               call.channel_id)
                await self._end_call(call.channel_id, hangup=True)

    def _attach_media(self, uuid_payload: bytes,
                     writer: asyncio.StreamWriter) -> Optional[_Call]:
        """Map a freshly-connected AudioSocket to its call via the UUID frame."""
        try:
            if len(uuid_payload) == 16:
                media_uuid = str(uuid_mod.UUID(bytes=uuid_payload))
            else:
                # Some builds send the UUID as an ASCII string instead of raw
                # bytes — normalise to canonical form for the lookup.
                media_uuid = str(uuid_mod.UUID(uuid_payload.decode("ascii", "ignore")))
        except Exception:
            logger.warning("SIP: unparseable AudioSocket UUID frame (%d bytes)",
                           len(uuid_payload))
            return None
        channel_id = self._uuid_to_channel.get(media_uuid)
        if not channel_id:
            logger.warning("SIP: AudioSocket UUID %s did not match any call", media_uuid)
            return None
        call = self._calls.get(channel_id)
        if call is not None:
            call.writer = writer
            call.synth_task = asyncio.create_task(self._synth_loop(call))
            call.playout_task = asyncio.create_task(self._playout_loop(call))
            logger.debug("SIP: media attached for call %s", channel_id)
            # Anything already queued (e.g. an outbound call's opening
            # message, queued at Call-construction time) starts draining now.
        return call

    def _on_caller_audio(self, call: _Call, pcm: bytes) -> None:
        if call.is_speaking():
            if not self.barge_in:
                # Half-duplex: while Hermes speaks, the inbound line carries
                # analog echo of our own TTS — feeding it to the VAD would
                # let the bot interrupt (and transcribe) itself.
                call.detector.reset()
                return
        utterance = call.detector.feed(pcm)
        if utterance is not None:
            if self.barge_in and call.is_speaking():
                self._flush_playback(call)
            task = asyncio.create_task(self._handle_utterance(call, utterance))
            # Hold a strong reference: asyncio keeps only weak refs to
            # tasks, and an unreferenced STT/agent task can be GC'd
            # mid-flight (the caller's turn would silently vanish).
            call.utterance_tasks.add(task)
            task.add_done_callback(call.utterance_tasks.discard)

    def _flush_playback(self, call: _Call) -> None:
        """Abort in-flight and queued reply audio (barge-in)."""
        call.flush_generation += 1
        for q in (call.say_queue, call.pcm_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    @staticmethod
    def _transcribe_pcm(pcm: bytes) -> Dict[str, Any]:
        """Blocking helper: write a temp WAV, run STT, clean up.

        Runs inside asyncio.to_thread so neither the file I/O nor the STT
        call touches the event loop that paces live audio.
        """
        from tools.transcription_tools import transcribe_audio

        wav_bytes = pcm_to_wav_bytes(pcm)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(wav_bytes)
            wav_path = tf.name
        try:
            return transcribe_audio(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    async def _handle_utterance(self, call: _Call, pcm: bytes) -> None:
        """STT a finished utterance and hand it to the agent loop."""
        if not self._message_handler:
            return
        try:
            result = await asyncio.to_thread(self._transcribe_pcm, pcm)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("SIP: transcription failed — %s", e)
            return

        text = (result or {}).get("transcript", "").strip()
        if not result or not result.get("success") or not text:
            logger.debug("SIP: empty/failed transcript for call %s", call.channel_id)
            return
        logger.info("SIP: call %s caller said: %s", call.channel_id, text)

        source = self.build_source(
            chat_id=call.channel_id,
            chat_name=f"Phone {call.caller}",
            chat_type="dm",
            user_id=call.caller,
            user_name=call.caller,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.VOICE,
            source=source,
            message_id=str(int(time.time() * 1000)),
        )
        await self.handle_message(event)

    # ── Sending: agent reply → speech ─────────────────────────────────────

    async def send(self, chat_id: str, content: str,
                   reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        call = self._calls.get(chat_id)
        if call is not None:
            if not content or not content.strip():
                return SendResult(success=True, message_id=str(int(time.time() * 1000)))
            # Enqueue — replies within a turn are spoken in order.  Barge-in
            # (not replacement) is the only thing that cancels speech.  Safe
            # even before media attaches: say_queue is a plain FIFO, and
            # _synth_loop starts draining it the moment AudioSocket connects.
            await call.say_queue.put(content)
            return SendResult(success=True, message_id=str(int(time.time() * 1000)))

        # No live call at this chat_id: this is how cron/send_message_tool
        # reach SIP (they call adapter.send(chat_id, content) on whatever
        # live adapter is registered, generically, for every platform).
        # Treat chat_id as a PJSIP endpoint name and ring it.
        return await self.originate_call(destination=chat_id, message=content)

    async def originate_call(self, destination: Optional[str] = None,
                             message: Optional[str] = None) -> SendResult:
        """Have Hermes call out to *destination* (a PJSIP endpoint name).

        Defaults to ``SIP_OUTBOUND_ENDPOINT`` (the same endpoint the OBi200
        registers as).  If *message* is given it is spoken as soon as the
        call is answered and media attaches; the call then continues as a
        normal live conversation until either side hangs up.
        """
        if self._session is None:
            return SendResult(success=False, error="SIP adapter is not connected to ARI")
        endpoint = (destination or self.outbound_endpoint or "").strip()
        if not endpoint:
            return SendResult(success=False, error="No destination endpoint configured")

        channel_id = f"sip-out-{uuid_mod.uuid4().hex}"
        self._pending_outbound[channel_id] = message

        result = await self._ari_request("POST", "channels", params={
            "endpoint": f"PJSIP/{endpoint}",
            "app": self.stasis_app,
            "appArgs": "outbound",
            "channelId": channel_id,
            "callerId": "Hermes",
            "timeout": self.outbound_timeout_seconds,
        })
        if not result or "id" not in result:
            self._pending_outbound.pop(channel_id, None)
            logger.error("SIP: failed to originate call to %s", redact_phone(endpoint))
            return SendResult(success=False,
                             error=f"Could not originate call to {endpoint}")
        logger.info("SIP: originating call %s to %s", channel_id, redact_phone(endpoint))
        return SendResult(success=True, message_id=channel_id)

    async def _synth_loop(self, call: _Call) -> None:
        """Stage 1: texts from say_queue → sentence TTS → PCM into pcm_queue.

        Runs for the call's lifetime.  Because this is a separate task from
        playout, sentence N+1 synthesizes while sentence N is still playing —
        no dead air between sentences beyond the first.
        """
        from tools.tts_tool import _strip_markdown_for_tts

        try:
            while True:
                text = await call.say_queue.get()
                generation = call.flush_generation
                spoken = _strip_markdown_for_tts(text)
                for sentence in _split_sentences(spoken):
                    if call.flush_generation != generation:
                        break  # barge-in flushed this reply
                    pcm = await self._synthesize_slin(sentence)
                    if not pcm or call.flush_generation != generation:
                        continue
                    await call.pcm_queue.put((generation, pcm))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("SIP: synth loop error on call %s — %s", call.channel_id, e)

    async def _playout_loop(self, call: _Call) -> None:
        """Stage 2: PCM chunks from pcm_queue → paced AudioSocket frames."""
        try:
            while True:
                generation, pcm = await call.pcm_queue.get()
                if generation != call.flush_generation:
                    continue  # stale audio flushed by barge-in
                call.playing = True
                try:
                    await self._stream_frames(call, pcm, generation)
                finally:
                    call.playing = False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("SIP: playout loop error on call %s — %s", call.channel_id, e)

    async def _synthesize_slin(self, text: str) -> bytes:
        """TTS *text* and transcode to 8 kHz mono signed-linear PCM."""
        from tools.tts_tool import text_to_speech_tool

        tmp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_audio.close()
        produced_path = tmp_audio.name
        try:
            res = await asyncio.to_thread(text_to_speech_tool, text, tmp_audio.name)
            try:
                parsed = json.loads(res) if isinstance(res, str) else res
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                if parsed.get("success") is False:
                    logger.warning("SIP: TTS failed: %s", parsed.get("error"))
                    return b""
                # Command/plugin TTS providers may write to a different path
                # (e.g. the extension rewritten per output_format); the JSON
                # carries the real location.
                real = parsed.get("file_path")
                if isinstance(real, str) and real:
                    produced_path = real
            if not os.path.exists(produced_path) or os.path.getsize(produced_path) == 0:
                logger.warning("SIP: TTS produced no audio for sentence")
                return b""
            return await _ffmpeg_to_slin(produced_path)
        finally:
            for path in {tmp_audio.name, produced_path}:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    async def _stream_frames(self, call: _Call, pcm: bytes,
                             generation: Optional[int] = None) -> None:
        """Write 20 ms SLIN frames to the AudioSocket with real-time pacing."""
        writer = call.writer
        if writer is None:
            return
        view = memoryview(pcm)
        next_t = time.monotonic()
        for i in range(0, len(pcm), SLIN_FRAME_BYTES):
            if writer.is_closing():
                return
            if generation is not None and generation != call.flush_generation:
                return  # barge-in: stop mid-chunk
            frame = view[i : i + SLIN_FRAME_BYTES]
            if len(frame) < SLIN_FRAME_BYTES:
                writer.write(encode_audiosocket_frame(AS_KIND_SLIN, bytes(frame)
                             + b"\x00" * (SLIN_FRAME_BYTES - len(frame))))
            else:
                writer.write(_SLIN_FRAME_HEADER)
                writer.write(frame)
            await writer.drain()
            next_t += SLIN_FRAME_MS / 1000.0
            delay = next_t - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    # ── Media senders ─────────────────────────────────────────────────────
    #
    # The base-class defaults for these send fallback TEXT ("⚠️ Couldn't
    # deliver the audio attachment.", raw image URLs) through send(), which
    # a phone caller would hear read aloud.  Voice attachments are playable;
    # everything else speaks its caption (real content) and skips the file.

    async def send_voice(self, chat_id: str, audio_path: str,
                         caption: Optional[str] = None,
                         reply_to: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         **kwargs) -> SendResult:
        call = self._calls.get(chat_id)
        if call is None or call.writer is None:
            return SendResult(success=False, error="No active call media for chat_id")
        pcm = await _ffmpeg_to_slin(audio_path)
        if not pcm:
            return SendResult(success=False, error="Could not decode audio for playback")
        await call.pcm_queue.put((call.flush_generation, pcm))
        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def _speak_caption_only(self, chat_id: str,
                                  caption: Optional[str]) -> SendResult:
        if caption and caption.strip():
            return await self.send(chat_id, caption)
        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def send_image(self, chat_id: str, image_url: str,
                         caption: Optional[str] = None, reply_to=None,
                         metadata=None, **kwargs) -> SendResult:
        return await self._speak_caption_only(chat_id, caption)

    async def send_image_file(self, chat_id: str, image_path: str,
                              caption: Optional[str] = None, reply_to=None,
                              metadata=None, **kwargs) -> SendResult:
        return await self._speak_caption_only(chat_id, caption)

    async def send_video(self, chat_id: str, video_path: str,
                         caption: Optional[str] = None, reply_to=None,
                         metadata=None, **kwargs) -> SendResult:
        return await self._speak_caption_only(chat_id, caption)

    async def send_animation(self, chat_id: str, animation_path: str,
                             caption: Optional[str] = None, reply_to=None,
                             metadata=None, **kwargs) -> SendResult:
        return await self._speak_caption_only(chat_id, caption)

    async def send_document(self, chat_id: str, document_path: str,
                            caption: Optional[str] = None, reply_to=None,
                            metadata=None, **kwargs) -> SendResult:
        return await self._speak_caption_only(chat_id, caption)

    def _should_auto_tts_for_chat(self, chat_id: str) -> bool:
        # Every SIP reply is already spoken by send(); the gateway's
        # auto-TTS would synthesize a second, redundant audio file per turn.
        return False

    # ── Misc required hooks ───────────────────────────────────────────────

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None  # No typing indicator on a phone call.

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        call = self._calls.get(chat_id)
        caller = call.caller if call else chat_id
        return {"name": f"Phone {caller}", "type": "dm", "chat_id": chat_id}


# ---------------------------------------------------------------------------
# Audio helpers (module-level so they are easy to test / mock)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentence-ish chunks to lower playback latency.

    Uses the same boundary regex as the streaming TTS pipeline in
    tools/tts_tool.py so tuning fixes there reach the phone path too.
    """
    from tools.tts_tool import _SENTENCE_BOUNDARY_RE

    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _resolve_ffmpeg() -> str:
    """Resolve the ffmpeg binary the way the rest of the codebase does
    (Homebrew/local prefixes before PATH), falling back to plain 'ffmpeg'."""
    try:
        from tools.transcription_tools import _find_ffmpeg_binary
        found = _find_ffmpeg_binary()
        if found:
            return found
    except Exception:
        pass
    return "ffmpeg"


async def _ffmpeg_to_slin(path: str) -> bytes:
    """Decode an audio file to raw 8 kHz mono s16le PCM via ffmpeg."""
    cmd = [
        _resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1",
        "-ar", str(SLIN_SAMPLE_RATE), "-",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning("SIP: ffmpeg transcode failed: %s",
                       stderr.decode("utf-8", "ignore")[:200])
        return b""
    return stdout


# ---------------------------------------------------------------------------
# Plugin registration hooks
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """True when the SIP bridge is minimally configured and aiohttp present."""
    if not (os.getenv("SIP_ARI_URL") and os.getenv("SIP_ARI_USER")
            and os.getenv("SIP_ARI_PASSWORD")):
        return False
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        return False


def validate_config(config) -> bool:
    """True when the platform config carries the required ARI settings."""
    return _ari_configured(getattr(config, "extra", {}) or {})


# Same predicate: "configured" is the best connectivity signal available
# without instantiating the adapter.
is_connected = validate_config


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load."""
    url = os.getenv("SIP_ARI_URL", "").strip()
    user = os.getenv("SIP_ARI_USER", "").strip()
    password = os.getenv("SIP_ARI_PASSWORD", "").strip()
    if not (url and user and password):
        return None
    seed: dict = {"ari_url": url, "ari_user": user, "ari_password": password}
    for env, key in (
        ("SIP_STASIS_APP", "stasis_app"),
        ("SIP_AUDIOSOCKET_HOST", "audiosocket_host"),
        ("SIP_AUDIOSOCKET_PORT", "audiosocket_port"),
        ("SIP_AUDIOSOCKET_ADVERTISE_HOST", "audiosocket_advertise_host"),
        ("SIP_BARGE_IN", "barge_in"),
        ("SIP_VAD_SILENCE_RMS", "vad_silence_rms"),
        ("SIP_VAD_SILENCE_SECONDS", "vad_silence_seconds"),
        ("SIP_OUTBOUND_ENDPOINT", "outbound_endpoint"),
        ("SIP_OUTBOUND_TIMEOUT_SECONDS", "outbound_timeout_seconds"),
    ):
        val = os.getenv(env, "").strip()
        if val:
            seed[key] = val

    # Default cron/send_message destination: the same endpoint the OBi200
    # registers as, so `cronjob(deliver="sip", ...)` rings the phone without
    # needing an explicit chat_id.  SIP_HOME_CHANNEL overrides it (e.g. a
    # second registered endpoint used only for outbound notifications).
    outbound_endpoint = os.getenv("SIP_OUTBOUND_ENDPOINT", "").strip() or "obi200"
    home = os.getenv("SIP_HOME_CHANNEL", "").strip() or outbound_endpoint
    seed["home_channel"] = {"chat_id": home, "name": f"Phone ({home})"}
    return seed


def interactive_setup() -> None:
    """`hermes gateway setup` flow for the SIP bridge."""
    from hermes_cli.setup import (
        prompt, prompt_yes_no, save_env_value, get_env_value,
        print_header, print_info, print_warning, print_success,
    )

    print_header("SIP voice bridge (Asterisk + OBi200)")
    print_info("Bridge an analog phone (via a SIP ATA like the OBi200) to Hermes.")
    print_info("Requires an Asterisk PBX with ARI enabled and a Stasis dialplan.")
    print_info("See plugins/platforms/sip/asterisk/README.md for the PBX wiring.")
    print()

    if get_env_value("SIP_ARI_URL") and not prompt_yes_no("Reconfigure SIP bridge?", False):
        return

    url = prompt("Asterisk ARI base URL", default=get_env_value("SIP_ARI_URL") or "http://127.0.0.1:8088")
    if not url:
        print_warning("ARI URL is required — skipping SIP setup")
        return
    save_env_value("SIP_ARI_URL", url.strip())

    user = prompt("ARI username", default=get_env_value("SIP_ARI_USER") or "")
    if not user:
        print_warning("ARI username is required — skipping SIP setup")
        return
    save_env_value("SIP_ARI_USER", user.strip())

    password = prompt("ARI password", password=True)
    if password:
        save_env_value("SIP_ARI_PASSWORD", password)

    app = prompt("Stasis app name", default=get_env_value("SIP_STASIS_APP") or "hermes")
    save_env_value("SIP_STASIS_APP", (app or "hermes").strip())

    advertise = prompt(
        "Address Asterisk dials back for media (host reachable from the PBX)",
        default=get_env_value("SIP_AUDIOSOCKET_ADVERTISE_HOST") or "127.0.0.1")
    save_env_value("SIP_AUDIOSOCKET_ADVERTISE_HOST", (advertise or "127.0.0.1").strip())

    port = prompt("AudioSocket TCP port",
                  default=get_env_value("SIP_AUDIOSOCKET_PORT") or "9092")
    if port:
        try:
            save_env_value("SIP_AUDIOSOCKET_PORT", str(int(port)))
        except ValueError:
            print_warning("Invalid port — keeping default 9092")

    print()
    print_info("🔒 Access control: which caller numbers may talk to Hermes")
    if prompt_yes_no("Allow any caller (dev only)?", False):
        save_env_value("SIP_ALLOW_ALL_USERS", "true")
        save_env_value("SIP_ALLOWED_USERS", "")
        print_warning("⚠️  Open access — anyone who can reach the PBX can talk to Hermes.")
    else:
        save_env_value("SIP_ALLOW_ALL_USERS", "false")
        allowed = prompt("Allowed caller numbers (comma-separated)",
                         default=get_env_value("SIP_ALLOWED_USERS") or "")
        save_env_value("SIP_ALLOWED_USERS", (allowed or "").replace(" ", ""))

    print()
    print_success("SIP bridge configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


_PLATFORM_HINT = (
    "You are talking to the user on a phone call (analog handset bridged over "
    "SIP). Your replies are read aloud by text-to-speech, so: keep responses "
    "short and conversational, use plain spoken language with no markdown, "
    "lists, code blocks, emoji, or URLs, and prefer one or two sentences. If "
    "something needs a long answer, give the short version and offer to "
    "continue. Spell out things that must be heard clearly (numbers, codes)."
)


def register(ctx):
    """Plugin entry point: called by the Hermes plugin system."""
    ctx.register_platform(
        name="sip",
        label="SIP",
        adapter_factory=lambda cfg: SIPAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["SIP_ARI_URL", "SIP_ARI_USER", "SIP_ARI_PASSWORD"],
        install_hint="pip install hermes-agent[sip] and run an Asterisk PBX with ARI",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        allowed_users_env="SIP_ALLOWED_USERS",
        allow_all_env="SIP_ALLOW_ALL_USERS",
        emoji="☎️",
        # Callers are identified by phone number: redact before the LLM,
        # matching WhatsApp/Signal/BlueBubbles behavior.
        pii_safe=True,
        platform_hint=_PLATFORM_HINT,
        # `cronjob(..., deliver="sip")` / `send_message_tool(platform="sip", ...)`
        # with no explicit chat_id rings SIP_HOME_CHANNEL (default:
        # SIP_OUTBOUND_ENDPOINT) — see originate_call() and _env_enablement().
        # No standalone_sender_fn: origination needs the live ARI connection
        # and AudioSocket server the running gateway process owns; there is
        # no meaningful out-of-process equivalent.
        cron_deliver_env_var="SIP_HOME_CHANNEL",
    )
