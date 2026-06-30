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

Audio flow::

    caller speaks ─▶ AudioSocket PCM ─▶ VAD turn detector ─▶ WAV
                  ─▶ transcribe_audio() ─▶ handle_message() (agent loop)
    agent reply  ─▶ send() ─▶ text_to_speech_tool() ─▶ ffmpeg 8k s16le
                  ─▶ 320-byte frames ─▶ AudioSocket

The adapter is a plugin: it subclasses ``BasePlatformAdapter`` and registers
via ``register(ctx)`` with zero changes to core Hermes code.  See
``asterisk/README.md`` in this directory for the PBX + OBi200 wiring.

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
    SIP_VAD_SILENCE_RMS         end-of-turn RMS threshold (default: 200)
    SIP_VAD_SILENCE_SECONDS     silence to end a turn      (default: 1.5)
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
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.config import Platform

logger = logging.getLogger(__name__)


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
    for i in range(0, len(pcm), frame_bytes):
        chunk = pcm[i : i + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
        out.append(chunk)
    return out


def pcm_rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of signed-linear 16-bit mono PCM.

    Uses numpy when available (fast path); falls back to pure Python so the
    VAD remains importable in minimal environments.
    """
    if not pcm:
        return 0.0
    try:
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples * samples)))
    except Exception:
        # Pure-Python fallback (also covers numpy-less test envs).
        count = len(pcm) // 2
        if count == 0:
            return 0.0
        total = 0
        for s in struct.unpack(f"<{count}h", pcm[: count * 2]):
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
    """

    def __init__(
        self,
        *,
        sample_rate: int = SLIN_SAMPLE_RATE,
        silence_rms: float = 200.0,
        silence_seconds: float = 1.5,
        min_speech_seconds: float = 0.3,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_rms = silence_rms
        self.silence_seconds = silence_seconds
        self.min_speech_seconds = min_speech_seconds
        self.reset()

    def reset(self) -> None:
        self._buf = bytearray()
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0
        self._in_speech = False

    def _frame_seconds(self, pcm: bytes) -> float:
        samples = len(pcm) // 2
        return samples / float(self.sample_rate) if self.sample_rate else 0.0

    def feed(self, pcm: bytes) -> Optional[bytes]:
        """Process one audio frame.  Returns a finished utterance or None."""
        if not pcm:
            return None
        dur = self._frame_seconds(pcm)
        rms = pcm_rms(pcm)
        is_speech = rms >= self.silence_rms

        if is_speech:
            self._speech_seconds += dur
            self._silence_seconds = 0.0
            if self._speech_seconds >= self.min_speech_seconds:
                self._in_speech = True
            # Always buffer once we are confidently in speech.
            if self._in_speech:
                self._buf.extend(pcm)
            return None

        # Silence frame.
        if self._in_speech:
            self._buf.extend(pcm)  # keep trailing silence inside the utterance
            self._silence_seconds += dur
            if self._silence_seconds >= self.silence_seconds:
                utterance = bytes(self._buf)
                self.reset()
                return utterance
        else:
            # Not yet confidently in speech — decay the speech counter so a
            # brief blip below min_speech does not eventually trigger.
            self._speech_seconds = 0.0
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


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Per-call state
# ---------------------------------------------------------------------------

class _Call:
    """Bookkeeping for one in-flight phone call."""

    def __init__(self, channel_id: str, caller: str) -> None:
        self.channel_id = channel_id
        self.caller = caller or "unknown"
        # Canonical UUID string (8-4-4-4-12).  Asterisk's externalMedia ``data``
        # field expects a valid UUID and the AudioSocket ID frame echoes it back
        # as 16 raw bytes, which we re-render to this same canonical form.
        self.media_uuid = str(uuid_mod.uuid4())
        self.bridge_id: Optional[str] = None
        self.external_channel_id: Optional[str] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.detector = PhoneTurnDetector()
        self.playback_task: Optional[asyncio.Task] = None
        self.recv_buf = b""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class SIPAdapter(BasePlatformAdapter):
    """SIP voice bridge adapter (Asterisk ARI + AudioSocket)."""

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
                extra, "SIP_VAD_SILENCE_RMS", "vad_silence_rms", 200.0))
        except (TypeError, ValueError):
            self.vad_silence_rms = 200.0
        try:
            self.vad_silence_seconds = float(_env_or_extra(
                extra, "SIP_VAD_SILENCE_SECONDS", "vad_silence_seconds", 1.5))
        except (TypeError, ValueError):
            self.vad_silence_seconds = 1.5

        # Runtime state
        self._session = None  # aiohttp.ClientSession
        self._ws = None       # aiohttp ARI event websocket
        self._ws_task: Optional[asyncio.Task] = None
        self._server: Optional[asyncio.AbstractServer] = None
        self._calls: Dict[str, _Call] = {}            # channel_id -> _Call
        self._uuid_to_channel: Dict[str, str] = {}    # media_uuid -> channel_id
        self._external_channels: set = set()          # externalMedia channel ids

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

        self._session = aiohttp.ClientSession(
            auth=aiohttp.BasicAuth(self.ari_user, self.ari_password))

        # Open the ARI event websocket subscribed to our Stasis app.
        ws_base = self.ari_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = (f"{ws_base}/ari/events?app={self.stasis_app}"
                  f"&api_key={self.ari_user}:{self.ari_password}&subscribeAll=true")
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
                           params: Optional[dict] = None) -> Any:
        """Issue an ARI REST request; returns parsed JSON or None."""
        if self._session is None:
            return None
        url = f"{self.ari_url}/ari/{path.lstrip('/')}"
        try:
            async with self._session.request(method, url, params=params) as resp:
                if resp.status >= 400:
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
            self._external_channels.discard(cid)

    async def _on_stasis_start(self, event: dict) -> None:
        channel = event.get("channel", {})
        channel_id = channel.get("id")
        if not channel_id:
            return
        # Ignore the externalMedia channel re-entering Stasis.
        if channel_id in self._external_channels:
            return
        name = channel.get("name", "")
        if name.startswith("AudioSocket") or name.startswith("UnicastRTP"):
            self._external_channels.add(channel_id)
            return

        caller = (channel.get("caller") or {}).get("number") or ""
        call = _Call(channel_id, caller)
        self._calls[channel_id] = call
        self._uuid_to_channel[call.media_uuid] = channel_id

        logger.info("SIP: incoming call %s from %s", channel_id, caller or "unknown")

        # Answer, then bridge the caller with an externalMedia (AudioSocket) leg.
        await self._ari_request("POST", f"channels/{channel_id}/answer")

        ext = await self._ari_request("POST", "channels/externalMedia", params={
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
        call.external_channel_id = ext["id"]
        self._external_channels.add(ext["id"])

        bridge = await self._ari_request("POST", "bridges", params={"type": "mixing"})
        if not bridge or "id" not in bridge:
            logger.error("SIP: bridge creation failed for %s", channel_id)
            await self._end_call(channel_id, hangup=True)
            return
        call.bridge_id = bridge["id"]
        await self._ari_request(
            "POST", f"bridges/{call.bridge_id}/addChannel",
            params={"channel": f"{channel_id},{call.external_channel_id}"})

    async def _end_call(self, channel_id: str, *, hangup: bool) -> None:
        call = self._calls.pop(channel_id, None)
        if call is None:
            return
        self._uuid_to_channel.pop(call.media_uuid, None)
        if call.playback_task and not call.playback_task.done():
            call.playback_task.cancel()
        if call.writer is not None:
            try:
                call.writer.close()
            except Exception:
                pass
        if call.bridge_id:
            await self._ari_request("DELETE", f"bridges/{call.bridge_id}")
        if hangup:
            await self._ari_request("DELETE", f"channels/{channel_id}")
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
                            await self._end_call(call.channel_id, hangup=False)
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
            call.detector = PhoneTurnDetector(
                silence_rms=self.vad_silence_rms,
                silence_seconds=self.vad_silence_seconds)
            logger.debug("SIP: media attached for call %s", channel_id)
        return call

    def _on_caller_audio(self, call: _Call, pcm: bytes) -> None:
        utterance = call.detector.feed(pcm)
        if utterance is not None:
            # Barge-in: a new utterance cancels any in-progress reply.
            if call.playback_task and not call.playback_task.done():
                call.playback_task.cancel()
            asyncio.create_task(self._handle_utterance(call, utterance))

    async def _handle_utterance(self, call: _Call, pcm: bytes) -> None:
        """STT a finished utterance and hand it to the agent loop."""
        if not self._message_handler:
            return
        wav_bytes = pcm_to_wav_bytes(pcm)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(wav_bytes)
            wav_path = tf.name
        try:
            from tools.transcription_tools import transcribe_audio
            result = await asyncio.to_thread(transcribe_audio, wav_path)
        except Exception as e:
            logger.warning("SIP: transcription failed — %s", e)
            return
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

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
        if call is None or call.writer is None:
            return SendResult(success=False, error="No active call media for chat_id")
        if not content or not content.strip():
            return SendResult(success=True, message_id=str(int(time.time() * 1000)))

        # Speak the reply on the call's media socket.  Replace any in-flight
        # playback so the latest reply wins.
        if call.playback_task and not call.playback_task.done():
            call.playback_task.cancel()
        call.playback_task = asyncio.create_task(self._speak(call, content))
        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def _speak(self, call: _Call, text: str) -> None:
        try:
            for sentence in _split_sentences(text):
                pcm = await self._synthesize_slin(sentence)
                if not pcm:
                    continue
                await self._stream_frames(call, pcm)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("SIP: playback error on call %s — %s", call.channel_id, e)

    async def _synthesize_slin(self, text: str) -> bytes:
        """TTS *text* and transcode to 8 kHz mono signed-linear PCM."""
        from tools.tts_tool import text_to_speech_tool

        tmp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_audio.close()
        try:
            res = await asyncio.to_thread(text_to_speech_tool, text, tmp_audio.name)
            try:
                parsed = json.loads(res) if isinstance(res, str) else res
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and parsed.get("success") is False:
                logger.warning("SIP: TTS failed: %s", parsed.get("error"))
                return b""
            if not os.path.exists(tmp_audio.name) or os.path.getsize(tmp_audio.name) == 0:
                return b""
            return await _ffmpeg_to_slin(tmp_audio.name)
        finally:
            try:
                os.unlink(tmp_audio.name)
            except OSError:
                pass

    async def _stream_frames(self, call: _Call, pcm: bytes) -> None:
        """Write 20 ms SLIN frames to the AudioSocket with real-time pacing."""
        writer = call.writer
        if writer is None:
            return
        next_t = time.monotonic()
        for frame in frame_pcm(pcm):
            if writer.is_closing():
                return
            writer.write(encode_audiosocket_frame(AS_KIND_SLIN, frame))
            await writer.drain()
            next_t += SLIN_FRAME_MS / 1000.0
            delay = next_t - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

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
    """Split *text* into sentence-ish chunks to lower playback latency."""
    import re

    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


async def _ffmpeg_to_slin(path: str) -> bytes:
    """Decode an audio file to raw 8 kHz mono s16le PCM via ffmpeg."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
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
    extra = getattr(config, "extra", {}) or {}
    return bool(
        _env_or_extra(extra, "SIP_ARI_URL", "ari_url")
        and _env_or_extra(extra, "SIP_ARI_USER", "ari_user")
        and _env_or_extra(extra, "SIP_ARI_PASSWORD", "ari_password")
    )


def is_connected(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(
        _env_or_extra(extra, "SIP_ARI_URL", "ari_url")
        and _env_or_extra(extra, "SIP_ARI_USER", "ari_user")
        and _env_or_extra(extra, "SIP_ARI_PASSWORD", "ari_password")
    )


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
        ("SIP_VAD_SILENCE_RMS", "vad_silence_rms"),
        ("SIP_VAD_SILENCE_SECONDS", "vad_silence_seconds"),
    ):
        val = os.getenv(env, "").strip()
        if val:
            seed[key] = val
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
        pii_safe=False,
        platform_hint=_PLATFORM_HINT,
    )
