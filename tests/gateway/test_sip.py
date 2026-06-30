"""Tests for the SIP voice-bridge platform adapter plugin.

Two tiers of coverage:

* **Pure helpers** (no I/O): AudioSocket framing, the VAD turn detector, PCM
  framing/RMS, WAV wrapping, sentence splitting, config/env parsing, and
  plugin registration.
* **Async orchestration** with the network boundary mocked: ARI REST/event
  dispatch, StasisStart call setup, the AudioSocket connection handler, the
  utterance → STT → agent hop, and the agent-reply → TTS → frame playback hop.

Only the genuinely external surface (a live Asterisk PBX speaking real
RTP/SIP, plus the audio fidelity of STT/TTS) is left to the manual end-to-end
check documented in plugins/platforms/sip/asterisk/README.md.
"""

import asyncio
import struct
import uuid as uuid_mod

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_sip = load_plugin_adapter("sip")

parse_audiosocket_frames = _sip.parse_audiosocket_frames
encode_audiosocket_frame = _sip.encode_audiosocket_frame
frame_pcm = _sip.frame_pcm
pcm_rms = _sip.pcm_rms
pcm_to_wav_bytes = _sip.pcm_to_wav_bytes
PhoneTurnDetector = _sip.PhoneTurnDetector
SIPAdapter = _sip.SIPAdapter
register = _sip.register
check_requirements = _sip.check_requirements
validate_config = _sip.validate_config
_env_enablement = _sip._env_enablement
_split_sentences = _sip._split_sentences

AS_KIND_UUID = _sip.AS_KIND_UUID
AS_KIND_SLIN = _sip.AS_KIND_SLIN
AS_KIND_HANGUP = _sip.AS_KIND_HANGUP
AS_KIND_ERROR = _sip.AS_KIND_ERROR
SLIN_FRAME_BYTES = _sip.SLIN_FRAME_BYTES
_Call = _sip._Call
MessageType = _sip.MessageType


def _make_adapter(**env):
    """Construct a SIPAdapter with a minimal PlatformConfig and clean env."""
    from gateway.config import PlatformConfig

    return SIPAdapter(PlatformConfig(enabled=True, extra=dict(env)))


def _pcm(amplitude: int, n_samples: int) -> bytes:
    """Build n_samples of constant-amplitude signed-linear 16-bit PCM."""
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _frame(amplitude: int) -> bytes:
    """One 20 ms / 160-sample SLIN frame at the given amplitude."""
    return _pcm(amplitude, SLIN_FRAME_BYTES // 2)


# ── AudioSocket framing ────────────────────────────────────────────────────

class TestAudioSocketFraming:

    def test_roundtrip_single_frame(self):
        payload = b"\x01\x02\x03\x04"
        wire = encode_audiosocket_frame(AS_KIND_SLIN, payload)
        frames, rem = parse_audiosocket_frames(wire)
        assert rem == b""
        assert frames == [(AS_KIND_SLIN, payload)]

    def test_multiple_frames_in_one_buffer(self):
        wire = (encode_audiosocket_frame(AS_KIND_UUID, b"u" * 16)
                + encode_audiosocket_frame(AS_KIND_SLIN, b"audio")
                + encode_audiosocket_frame(AS_KIND_HANGUP, b""))
        frames, rem = parse_audiosocket_frames(wire)
        assert rem == b""
        assert [k for k, _ in frames] == [AS_KIND_UUID, AS_KIND_SLIN, AS_KIND_HANGUP]

    def test_partial_frame_held_as_remainder(self):
        full = encode_audiosocket_frame(AS_KIND_SLIN, b"hello world")
        # Feed only the header + part of the payload.
        head = full[:7]
        frames, rem = parse_audiosocket_frames(head)
        assert frames == []
        assert rem == head
        # Now feed the rest — reassembly across reads must work.
        frames, rem = parse_audiosocket_frames(rem + full[7:])
        assert rem == b""
        assert frames == [(AS_KIND_SLIN, b"hello world")]

    def test_header_split_across_reads(self):
        full = encode_audiosocket_frame(AS_KIND_SLIN, b"abc")
        frames, rem = parse_audiosocket_frames(full[:2])  # less than 3-byte header
        assert frames == []
        assert rem == full[:2]
        frames, rem = parse_audiosocket_frames(rem + full[2:])
        assert frames == [(AS_KIND_SLIN, b"abc")]

    def test_encode_length_field_is_big_endian(self):
        wire = encode_audiosocket_frame(AS_KIND_SLIN, b"\x00" * 320)
        assert wire[0] == AS_KIND_SLIN
        assert struct.unpack(">H", wire[1:3])[0] == 320

    def test_oversized_payload_rejected(self):
        with pytest.raises(ValueError):
            encode_audiosocket_frame(AS_KIND_SLIN, b"\x00" * 70000)


# ── PCM framing & RMS ──────────────────────────────────────────────────────

class TestPcmHelpers:

    def test_frame_pcm_pads_final_frame(self):
        pcm = b"\x01" * (SLIN_FRAME_BYTES + 10)
        frames = frame_pcm(pcm)
        assert len(frames) == 2
        assert all(len(f) == SLIN_FRAME_BYTES for f in frames)
        # Tail is zero-padded.
        assert frames[1][:10] == b"\x01" * 10
        assert frames[1][10:] == b"\x00" * (SLIN_FRAME_BYTES - 10)

    def test_frame_pcm_exact_multiple(self):
        pcm = b"\x02" * (SLIN_FRAME_BYTES * 3)
        frames = frame_pcm(pcm)
        assert len(frames) == 3
        assert all(len(f) == SLIN_FRAME_BYTES for f in frames)

    def test_frame_pcm_empty(self):
        assert frame_pcm(b"") == []

    def test_rms_silence_is_zero(self):
        assert pcm_rms(_pcm(0, 160)) == 0.0

    def test_rms_constant_amplitude(self):
        assert pcm_rms(_pcm(1000, 160)) == pytest.approx(1000.0, rel=1e-6)

    def test_rms_empty(self):
        assert pcm_rms(b"") == 0.0


# ── VAD turn detector ──────────────────────────────────────────────────────

class TestPhoneTurnDetector:

    def _run(self, det, speech_frames, silence_frames, amp=5000):
        emitted = None
        for _ in range(speech_frames):
            out = det.feed(_frame(amp))
            emitted = emitted or out
        for _ in range(silence_frames):
            out = det.feed(_frame(0))
            if out is not None:
                emitted = out
        return emitted

    def test_speech_then_silence_emits_one_utterance(self):
        det = PhoneTurnDetector(silence_rms=200, silence_seconds=1.5,
                                min_speech_seconds=0.3)
        # 20 speech frames (0.4s > 0.3s) then 80 silence frames (1.6s > 1.5s).
        utterance = self._run(det, speech_frames=20, silence_frames=80)
        assert utterance is not None
        # The buffered utterance contains the speech (and trailing silence).
        assert len(utterance) >= 20 * SLIN_FRAME_BYTES
        # Detector resets after emitting.
        assert det.feed(_frame(0)) is None

    def test_short_blip_is_ignored(self):
        det = PhoneTurnDetector(silence_rms=200, silence_seconds=1.5,
                                min_speech_seconds=0.3)
        # 5 speech frames = 0.1s < 0.3s confirm threshold → never a turn.
        utterance = self._run(det, speech_frames=5, silence_frames=100)
        assert utterance is None

    def test_silence_threshold_respected(self):
        # Amplitude below the threshold counts as silence, so no turn fires.
        det = PhoneTurnDetector(silence_rms=2000, silence_seconds=1.0,
                                min_speech_seconds=0.3)
        utterance = self._run(det, speech_frames=30, silence_frames=80, amp=500)
        assert utterance is None

    def test_does_not_emit_before_silence_elapses(self):
        det = PhoneTurnDetector(silence_rms=200, silence_seconds=1.5,
                                min_speech_seconds=0.3)
        # 20 speech + only 30 silence frames (0.6s < 1.5s) → not yet.
        utterance = self._run(det, speech_frames=20, silence_frames=30)
        assert utterance is None


# ── WAV wrapping ───────────────────────────────────────────────────────────

class TestWavWrap:

    def test_wav_header_is_8khz_mono_16bit(self):
        import io
        import wave

        data = pcm_to_wav_bytes(_pcm(123, 160))
        with wave.open(io.BytesIO(data), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 8000
            assert wf.getnframes() == 160


# ── Sentence splitting ─────────────────────────────────────────────────────

class TestSentenceSplit:

    def test_splits_on_sentence_boundaries(self):
        assert _split_sentences("Hi there. How are you? Good!") == [
            "Hi there.", "How are you?", "Good!"]

    def test_empty(self):
        assert _split_sentences("   ") == []


# ── Config / env parsing ───────────────────────────────────────────────────

class TestConfig:

    def test_adapter_defaults(self, monkeypatch):
        for var in ("SIP_ARI_URL", "SIP_ARI_USER", "SIP_ARI_PASSWORD",
                    "SIP_STASIS_APP", "SIP_AUDIOSOCKET_PORT",
                    "SIP_AUDIOSOCKET_ADVERTISE_HOST"):
            monkeypatch.delenv(var, raising=False)
        from gateway.config import PlatformConfig

        adapter = SIPAdapter(PlatformConfig(enabled=True))
        assert adapter.stasis_app == "hermes"
        assert adapter.bind_port == 9092
        assert adapter.advertise_host == "127.0.0.1"
        assert adapter.vad_silence_seconds == 1.5

    def test_env_overrides_extra(self, monkeypatch):
        monkeypatch.setenv("SIP_ARI_URL", "http://pbx:8088")
        monkeypatch.setenv("SIP_STASIS_APP", "envapp")
        from gateway.config import PlatformConfig

        cfg = PlatformConfig(enabled=True, extra={"stasis_app": "yamlapp",
                                                  "audiosocket_port": 7000})
        adapter = SIPAdapter(cfg)
        assert adapter.ari_url == "http://pbx:8088"
        assert adapter.stasis_app == "envapp"     # env wins over extra
        assert adapter.bind_port == 7000          # falls back to extra

    def test_env_enablement_requires_full_ari(self, monkeypatch):
        for var in ("SIP_ARI_URL", "SIP_ARI_USER", "SIP_ARI_PASSWORD"):
            monkeypatch.delenv(var, raising=False)
        assert _env_enablement() is None

        monkeypatch.setenv("SIP_ARI_URL", "http://pbx:8088")
        monkeypatch.setenv("SIP_ARI_USER", "hermes")
        monkeypatch.setenv("SIP_ARI_PASSWORD", "secret")
        monkeypatch.setenv("SIP_STASIS_APP", "hermes")
        seed = _env_enablement()
        assert seed["ari_url"] == "http://pbx:8088"
        assert seed["ari_user"] == "hermes"
        assert seed["stasis_app"] == "hermes"

    def test_validate_config(self, monkeypatch):
        for var in ("SIP_ARI_URL", "SIP_ARI_USER", "SIP_ARI_PASSWORD"):
            monkeypatch.delenv(var, raising=False)
        from gateway.config import PlatformConfig

        assert validate_config(PlatformConfig(enabled=True)) is False
        cfg = PlatformConfig(enabled=True, extra={
            "ari_url": "http://pbx:8088", "ari_user": "h", "ari_password": "p"})
        assert validate_config(cfg) is True


# ── Plugin registration ────────────────────────────────────────────────────

class TestRegistration:

    def test_register_calls_register_platform_with_sip(self):
        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs["name"] == "sip"
        assert kwargs["allowed_users_env"] == "SIP_ALLOWED_USERS"
        assert kwargs["allow_all_env"] == "SIP_ALLOW_ALL_USERS"
        assert "SIP_ARI_URL" in kwargs["required_env"]
        # No standalone sender — cron delivery to a live call is meaningless.
        assert "standalone_sender_fn" not in kwargs

    def test_check_requirements_needs_env(self, monkeypatch):
        for var in ("SIP_ARI_URL", "SIP_ARI_USER", "SIP_ARI_PASSWORD"):
            monkeypatch.delenv(var, raising=False)
        assert check_requirements() is False


# ── Test doubles for the network boundary ──────────────────────────────────

class _FakeWriter:
    """Minimal asyncio.StreamWriter stand-in capturing written bytes."""

    def __init__(self):
        self.buf = bytearray()
        self.closed = False

    def write(self, data):
        self.buf.extend(data)

    async def drain(self):
        return None

    def is_closing(self):
        return self.closed

    def close(self):
        self.closed = True


class _FakeReader:
    """Yields a scripted list of byte chunks, then EOF."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def read(self, _n):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _OneShotDetector:
    """Detector stub: emits a fixed utterance once after N feeds."""

    def __init__(self, utterance, after=1):
        self._utterance = utterance
        self._after = after
        self.feeds = 0

    def feed(self, pcm):
        self.feeds += 1
        if self.feeds == self._after:
            return self._utterance
        return None


def _ari_cm(payload, status=200, content_type="application/json"):
    """Build an async-context-manager mock mimicking aiohttp's response."""
    resp = MagicMock()
    resp.status = status
    resp.content_type = content_type
    resp.json = AsyncMock(return_value=payload)
    resp.text = AsyncMock(return_value=payload if isinstance(payload, str) else "")
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ── connect() guards ───────────────────────────────────────────────────────

class TestConnectGuards:

    @pytest.mark.asyncio
    async def test_connect_fails_without_ari_config(self, monkeypatch):
        for var in ("SIP_ARI_URL", "SIP_ARI_USER", "SIP_ARI_PASSWORD"):
            monkeypatch.delenv(var, raising=False)
        adapter = _make_adapter()
        assert await adapter.connect() is False
        assert not adapter.is_connected


# ── ARI REST + event dispatch ──────────────────────────────────────────────

class TestARI:

    @pytest.mark.asyncio
    async def test_ari_request_returns_json(self):
        adapter = _make_adapter(ari_url="http://pbx:8088", ari_user="u", ari_password="p")
        session = MagicMock()
        session.request = MagicMock(return_value=_ari_cm({"id": "x"}))
        adapter._session = session
        out = await adapter._ari_request("POST", "channels/x/answer")
        assert out == {"id": "x"}

    @pytest.mark.asyncio
    async def test_ari_request_error_status_returns_none(self):
        adapter = _make_adapter(ari_url="http://pbx:8088", ari_user="u", ari_password="p")
        session = MagicMock()
        session.request = MagicMock(return_value=_ari_cm("boom", status=500,
                                                         content_type="text/plain"))
        adapter._session = session
        assert await adapter._ari_request("DELETE", "bridges/none") is None

    @pytest.mark.asyncio
    async def test_dispatch_stasis_end_ends_known_call(self):
        adapter = _make_adapter()
        adapter._calls["chan1"] = _Call("chan1", "5551234")
        adapter._end_call = AsyncMock()
        await adapter._dispatch_ari_event(
            {"type": "StasisEnd", "channel": {"id": "chan1"}})
        adapter._end_call.assert_awaited_once_with("chan1", hangup=False)

    @pytest.mark.asyncio
    async def test_dispatch_stasis_start_routes(self):
        adapter = _make_adapter()
        adapter._on_stasis_start = AsyncMock()
        ev = {"type": "StasisStart", "channel": {"id": "c", "name": "PJSIP/obi200"}}
        await adapter._dispatch_ari_event(ev)
        adapter._on_stasis_start.assert_awaited_once_with(ev)


# ── StasisStart: full call setup ───────────────────────────────────────────

class TestStasisStart:

    @pytest.mark.asyncio
    async def test_happy_path_answers_bridges_and_maps_uuid(self):
        adapter = _make_adapter(audiosocket_advertise_host="10.0.0.5",
                                audiosocket_port=9092)
        calls = []

        async def fake_ari(method, path, params=None):
            calls.append((method, path, params))
            if path == "channels/externalMedia":
                return {"id": "ext-1"}
            if path == "bridges":
                return {"id": "br-1"}
            return None

        adapter._ari_request = AsyncMock(side_effect=fake_ari)
        await adapter._on_stasis_start(
            {"channel": {"id": "chan-A", "name": "PJSIP/obi200",
                         "caller": {"number": "5551234"}}})

        call = adapter._calls["chan-A"]
        assert call.caller == "5551234"
        assert call.external_channel_id == "ext-1"
        assert call.bridge_id == "br-1"
        # UUID mapped for the AudioSocket handshake.
        assert adapter._uuid_to_channel[call.media_uuid] == "chan-A"
        # externalMedia was asked for AudioSocket/TCP/slin with our advertise host.
        em = next(p for m, pth, p in calls if pth == "channels/externalMedia")
        assert em["encapsulation"] == "audiosocket"
        assert em["transport"] == "tcp"
        assert em["format"] == "slin"
        assert em["external_host"] == "10.0.0.5:9092"
        assert em["data"] == call.media_uuid
        # Both legs were added to the mixing bridge.
        add = next(p for m, pth, p in calls if pth.endswith("/addChannel"))
        assert add["channel"] == "chan-A,ext-1"

    @pytest.mark.asyncio
    async def test_external_media_failure_ends_call(self):
        adapter = _make_adapter()
        adapter._ari_request = AsyncMock(return_value=None)  # answer + externalMedia both None
        adapter._end_call = AsyncMock()
        await adapter._on_stasis_start(
            {"channel": {"id": "chan-B", "name": "PJSIP/obi200", "caller": {}}})
        adapter._end_call.assert_awaited_once_with("chan-B", hangup=True)

    @pytest.mark.asyncio
    async def test_ignores_external_media_channel_reentry(self):
        adapter = _make_adapter()
        adapter._external_channels.add("ext-x")
        adapter._ari_request = AsyncMock()
        await adapter._on_stasis_start({"channel": {"id": "ext-x", "name": "AudioSocket/..."}})
        adapter._ari_request.assert_not_called()
        assert "ext-x" not in adapter._calls

    @pytest.mark.asyncio
    async def test_ignores_audiosocket_named_channel(self):
        adapter = _make_adapter()
        adapter._ari_request = AsyncMock()
        await adapter._on_stasis_start({"channel": {"id": "as-1", "name": "AudioSocket/127.0.0.1"}})
        adapter._ari_request.assert_not_called()
        assert "as-1" in adapter._external_channels


# ── AudioSocket media path ─────────────────────────────────────────────────

class TestAudioSocketMedia:

    def test_attach_media_maps_uuid_and_arms_detector(self):
        adapter = _make_adapter()
        call = _Call("chan-1", "555")
        adapter._calls["chan-1"] = call
        adapter._uuid_to_channel[call.media_uuid] = "chan-1"
        writer = _FakeWriter()
        # Asterisk sends the UUID as 16 raw bytes.
        raw = uuid_mod.UUID(call.media_uuid).bytes
        attached = adapter._attach_media(raw, writer)
        assert attached is call
        assert call.writer is writer
        assert call.detector.silence_rms == adapter.vad_silence_rms

    def test_attach_media_unknown_uuid_returns_none(self):
        adapter = _make_adapter()
        raw = uuid_mod.uuid4().bytes
        assert adapter._attach_media(raw, _FakeWriter()) is None

    def test_attach_media_bad_payload_returns_none(self):
        adapter = _make_adapter()
        assert adapter._attach_media(b"not-a-uuid", _FakeWriter()) is None

    def test_on_caller_audio_dispatches_completed_utterance(self):
        adapter = _make_adapter()
        call = _Call("chan-1", "555")
        call.detector = _OneShotDetector(b"\x01\x02", after=1)
        seen = {}
        adapter._handle_utterance = AsyncMock(side_effect=lambda c, p: seen.update(pcm=p))

        async def drive():
            adapter._on_caller_audio(call, _frame(5000))
            await asyncio.sleep(0)  # let the scheduled task run
        asyncio.run(drive())
        assert seen.get("pcm") == b"\x01\x02"

    @pytest.mark.asyncio
    async def test_connection_handler_routes_uuid_audio_hangup(self):
        adapter = _make_adapter()
        call = _Call("chan-1", "555")
        adapter._calls["chan-1"] = call
        adapter._uuid_to_channel[call.media_uuid] = "chan-1"
        adapter._end_call = AsyncMock()
        routed = []
        adapter._on_caller_audio = lambda c, pcm: routed.append((c.channel_id, pcm))

        raw_uuid = uuid_mod.UUID(call.media_uuid).bytes
        chunks = [
            encode_audiosocket_frame(AS_KIND_UUID, raw_uuid),
            encode_audiosocket_frame(AS_KIND_SLIN, b"\x00" * 320),
            encode_audiosocket_frame(AS_KIND_ERROR, b"err"),
            encode_audiosocket_frame(AS_KIND_HANGUP, b""),
        ]
        writer = _FakeWriter()
        await adapter._handle_audiosocket_conn(_FakeReader(chunks), writer)

        assert call.writer is writer            # UUID frame attached the socket
        assert routed == [("chan-1", b"\x00" * 320)]
        adapter._end_call.assert_awaited_once_with("chan-1", hangup=False)
        assert writer.closed


# ── Utterance → STT → agent ────────────────────────────────────────────────

class TestUtteranceDispatch:

    @pytest.mark.asyncio
    async def test_transcribes_and_dispatches_message_event(self):
        adapter = _make_adapter()
        adapter._message_handler = AsyncMock()
        adapter.handle_message = AsyncMock()
        call = _Call("chan-1", "5551234")

        with patch("tools.transcription_tools.transcribe_audio",
                   return_value={"success": True, "transcript": "hello hermes"}):
            await adapter._handle_utterance(call, _pcm(5000, 160))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args.args[0]
        assert event.text == "hello hermes"
        assert event.message_type == MessageType.VOICE
        assert event.source.user_id == "5551234"

    @pytest.mark.asyncio
    async def test_empty_transcript_does_not_dispatch(self):
        adapter = _make_adapter()
        adapter._message_handler = AsyncMock()
        adapter.handle_message = AsyncMock()
        call = _Call("chan-1", "555")
        with patch("tools.transcription_tools.transcribe_audio",
                   return_value={"success": True, "transcript": "   "}):
            await adapter._handle_utterance(call, _pcm(5000, 160))
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_message_handler_is_noop(self):
        adapter = _make_adapter()
        adapter._message_handler = None
        call = _Call("chan-1", "555")
        # Must return without attempting transcription.
        await adapter._handle_utterance(call, _pcm(5000, 160))


# ── Agent reply → TTS → playback ───────────────────────────────────────────

class TestSendAndPlayback:

    @pytest.mark.asyncio
    async def test_send_without_active_call_fails(self):
        adapter = _make_adapter()
        res = await adapter.send("nope", "hi")
        assert res.success is False

    @pytest.mark.asyncio
    async def test_send_schedules_playback(self):
        adapter = _make_adapter()
        call = _Call("chan-1", "555")
        call.writer = _FakeWriter()
        adapter._calls["chan-1"] = call
        spoken = {}
        adapter._speak = AsyncMock(side_effect=lambda c, t: spoken.update(text=t))
        res = await adapter.send("chan-1", "Hello there.")
        assert res.success is True
        await call.playback_task
        assert spoken["text"] == "Hello there."

    @pytest.mark.asyncio
    async def test_send_empty_content_no_playback(self):
        adapter = _make_adapter()
        call = _Call("chan-1", "555")
        call.writer = _FakeWriter()
        adapter._calls["chan-1"] = call
        adapter._speak = AsyncMock()
        res = await adapter.send("chan-1", "   ")
        assert res.success is True
        adapter._speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak_synthesizes_each_sentence(self):
        adapter = _make_adapter()
        call = _Call("chan-1", "555")
        adapter._synthesize_slin = AsyncMock(return_value=b"\x00" * 320)
        adapter._stream_frames = AsyncMock()
        await adapter._speak(call, "First. Second!")
        assert adapter._synthesize_slin.await_count == 2
        assert adapter._stream_frames.await_count == 2

    @pytest.mark.asyncio
    async def test_stream_frames_writes_framed_slin(self):
        adapter = _make_adapter()
        call = _Call("chan-1", "555")
        call.writer = _FakeWriter()
        pcm = b"\x01" * (SLIN_FRAME_BYTES + 100)  # 2 frames after padding
        with patch.object(_sip.asyncio, "sleep", new=AsyncMock()):
            await adapter._stream_frames(call, pcm)
        frames, rem = parse_audiosocket_frames(bytes(call.writer.buf))
        assert rem == b""
        assert len(frames) == 2
        assert all(k == AS_KIND_SLIN and len(p) == SLIN_FRAME_BYTES for k, p in frames)

    @pytest.mark.asyncio
    async def test_synthesize_slin_runs_tts_then_ffmpeg(self):
        adapter = _make_adapter()

        def fake_tts(text, path):
            with open(path, "wb") as fh:
                fh.write(b"fake-mp3-bytes")
            return '{"success": true}'

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts), \
             patch.object(_sip, "_ffmpeg_to_slin", new=AsyncMock(return_value=b"PCMDATA")):
            out = await adapter._synthesize_slin("hello")
        assert out == b"PCMDATA"

    @pytest.mark.asyncio
    async def test_synthesize_slin_tts_failure_returns_empty(self):
        adapter = _make_adapter()
        with patch("tools.tts_tool.text_to_speech_tool",
                   side_effect=lambda t, p: '{"success": false, "error": "no key"}'):
            out = await adapter._synthesize_slin("hello")
        assert out == b""


# ── ffmpeg transcode boundary ──────────────────────────────────────────────

class TestFfmpeg:

    @pytest.mark.asyncio
    async def test_ffmpeg_success_returns_stdout(self):
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"rawpcm", b""))
        with patch.object(_sip.asyncio, "create_subprocess_exec",
                          new=AsyncMock(return_value=proc)):
            assert await _sip._ffmpeg_to_slin("/tmp/x.mp3") == b"rawpcm"

    @pytest.mark.asyncio
    async def test_ffmpeg_failure_returns_empty(self):
        proc = MagicMock()
        proc.returncode = 1
        proc.communicate = AsyncMock(return_value=(b"", b"boom"))
        with patch.object(_sip.asyncio, "create_subprocess_exec",
                          new=AsyncMock(return_value=proc)):
            assert await _sip._ffmpeg_to_slin("/tmp/x.mp3") == b""


# ── Lifecycle teardown ─────────────────────────────────────────────────────

class TestDisconnect:

    @pytest.mark.asyncio
    async def test_disconnect_ends_calls_and_clears_transports(self):
        adapter = _make_adapter()
        adapter._calls["chan-1"] = _Call("chan-1", "555")
        adapter._end_call = AsyncMock(side_effect=lambda cid, hangup: adapter._calls.pop(cid, None))
        await adapter.disconnect()
        adapter._end_call.assert_awaited()
        assert adapter._calls == {}
        assert not adapter.is_connected


# ── Misc hooks ─────────────────────────────────────────────────────────────

class TestMiscHooks:

    @pytest.mark.asyncio
    async def test_get_chat_info(self):
        adapter = _make_adapter()
        adapter._calls["chan-1"] = _Call("chan-1", "5551234")
        info = await adapter.get_chat_info("chan-1")
        assert info["type"] == "dm"
        assert "5551234" in info["name"]
        assert info["chat_id"] == "chan-1"

    @pytest.mark.asyncio
    async def test_send_typing_noop(self):
        adapter = _make_adapter()
        assert await adapter.send_typing("chan-1") is None


# ── numpy RMS fast path (covered when numpy is installed) ───────────────────

class TestNumpyRms:

    def test_numpy_rms_matches_pure_python(self):
        pytest.importorskip("numpy")
        sig = _pcm(1234, 160)
        assert pcm_rms(sig) == pytest.approx(1234.0, rel=1e-6)


# ── interactive_setup wizard ───────────────────────────────────────────────

class TestInteractiveSetup:

    def test_setup_saves_env(self, monkeypatch):
        import types

        saved = {}
        fake_setup = types.ModuleType("hermes_cli.setup")
        fake_setup.prompt = lambda *a, **k: k.get("default") or "filled"
        fake_setup.prompt_yes_no = lambda *a, **k: False
        fake_setup.save_env_value = lambda k, v: saved.__setitem__(k, v)
        fake_setup.get_env_value = lambda k: ""
        fake_setup.print_header = lambda *a, **k: None
        fake_setup.print_info = lambda *a, **k: None
        fake_setup.print_warning = lambda *a, **k: None
        fake_setup.print_success = lambda *a, **k: None
        monkeypatch.setitem(__import__("sys").modules, "hermes_cli.setup", fake_setup)

        _sip.interactive_setup()
        # The ARI URL prompt's default is the canned fallback when env is empty.
        assert saved.get("SIP_ARI_URL") == "http://127.0.0.1:8088"
        assert saved.get("SIP_STASIS_APP") == "hermes"
        # Declining "allow all" records the explicit deny flag.
        assert saved.get("SIP_ALLOW_ALL_USERS") == "false"


# ── connect() bring-up + ARI event loop ────────────────────────────────────

class TestConnectAndEventLoop:

    @pytest.mark.asyncio
    async def test_connect_brings_up_transports(self):
        adapter = _make_adapter(ari_url="http://pbx:8088", ari_user="u", ari_password="p")

        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()
        fake_ws = MagicMock()
        fake_ws.close = AsyncMock()
        fake_session = MagicMock()
        fake_session.ws_connect = AsyncMock(return_value=fake_ws)
        fake_session.close = AsyncMock()

        # Keep the event loop inert so connect()'s finally-path stays clean.
        adapter._ari_event_loop = AsyncMock()

        with patch.object(_sip.asyncio, "start_server",
                          new=AsyncMock(return_value=fake_server)), \
             patch("aiohttp.ClientSession", return_value=fake_session), \
             patch("aiohttp.BasicAuth", return_value=MagicMock()):
            ok = await adapter.connect()

        assert ok is True
        assert adapter.is_connected
        assert adapter._server is fake_server
        fake_session.ws_connect.assert_awaited_once()

        await adapter.disconnect()
        fake_ws.close.assert_awaited()
        fake_server.close.assert_called()

    @pytest.mark.asyncio
    async def test_connect_bind_failure_returns_false(self):
        adapter = _make_adapter(ari_url="http://pbx:8088", ari_user="u", ari_password="p")
        with patch.object(_sip.asyncio, "start_server",
                          new=AsyncMock(side_effect=OSError("addr in use"))):
            assert await adapter.connect() is False
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_event_loop_dispatches_text_then_exits_on_close(self):
        import aiohttp

        adapter = _make_adapter()

        text_msg = MagicMock(type=aiohttp.WSMsgType.TEXT,
                             data='{"type": "StasisEnd", "channel": {"id": "c"}}')
        close_msg = MagicMock(type=aiohttp.WSMsgType.CLOSED, data=None)

        class _WS:
            def __aiter__(self):
                return self

            def __init__(self):
                self._msgs = [text_msg, close_msg]

            async def __anext__(self):
                if self._msgs:
                    return self._msgs.pop(0)
                raise StopAsyncIteration

        adapter._ws = _WS()
        adapter._dispatch_ari_event = AsyncMock()
        # Not marked connected → the finally branch does not raise a fatal error.
        await adapter._ari_event_loop()
        adapter._dispatch_ari_event.assert_awaited_once()


# ── error / edge branches ──────────────────────────────────────────────────

class TestErrorBranches:

    @pytest.mark.asyncio
    async def test_handle_utterance_transcription_exception_is_swallowed(self):
        adapter = _make_adapter()
        adapter._message_handler = AsyncMock()
        adapter.handle_message = AsyncMock()
        call = _Call("chan-1", "555")
        with patch("tools.transcription_tools.transcribe_audio",
                   side_effect=RuntimeError("stt down")):
            await adapter._handle_utterance(call, _pcm(5000, 160))
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stream_frames_noop_without_writer(self):
        adapter = _make_adapter()
        call = _Call("chan-1", "555")  # no writer attached
        # Should return immediately without raising.
        await adapter._stream_frames(call, b"\x00" * SLIN_FRAME_BYTES)

    def test_setup_allow_all_path(self, monkeypatch):
        import types

        saved = {}
        fake_setup = types.ModuleType("hermes_cli.setup")
        fake_setup.prompt = lambda *a, **k: k.get("default") or "secret"
        fake_setup.prompt_yes_no = lambda *a, **k: True  # reconfigure + allow-all
        fake_setup.save_env_value = lambda k, v: saved.__setitem__(k, v)
        fake_setup.get_env_value = lambda k: "http://existing:8088"
        fake_setup.print_header = lambda *a, **k: None
        fake_setup.print_info = lambda *a, **k: None
        fake_setup.print_warning = lambda *a, **k: None
        fake_setup.print_success = lambda *a, **k: None
        monkeypatch.setitem(__import__("sys").modules, "hermes_cli.setup", fake_setup)

        _sip.interactive_setup()
        assert saved.get("SIP_ALLOW_ALL_USERS") == "true"
        assert saved.get("SIP_ALLOWED_USERS") == ""


# ── Defensive branches ─────────────────────────────────────────────────────

class TestDefensiveBranches:

    def test_pcm_rms_pure_python_fallback(self, monkeypatch):
        # Force the numpy fast path to raise so the pure-Python fallback runs.
        # Patch only numpy.frombuffer (leaving the module otherwise intact so
        # pytest's own numpy introspection in approx() keeps working).
        np = pytest.importorskip("numpy")

        def boom(*a, **k):
            raise RuntimeError("no np")

        monkeypatch.setattr(np, "frombuffer", boom)
        assert abs(pcm_rms(_pcm(1000, 160)) - 1000.0) < 0.01

    @pytest.mark.asyncio
    async def test_ari_request_without_session_returns_none(self):
        adapter = _make_adapter()
        adapter._session = None
        assert await adapter._ari_request("GET", "channels") is None

    @pytest.mark.asyncio
    async def test_ari_request_exception_returns_none(self):
        adapter = _make_adapter(ari_url="http://pbx:8088", ari_user="u", ari_password="p")
        session = MagicMock()
        session.request = MagicMock(side_effect=RuntimeError("conn reset"))
        adapter._session = session
        assert await adapter._ari_request("GET", "channels") is None

    @pytest.mark.asyncio
    async def test_connection_handler_swallows_read_error_and_closes(self):
        adapter = _make_adapter()

        class _BoomReader:
            async def read(self, _n):
                raise RuntimeError("socket exploded")

        writer = _FakeWriter()
        await adapter._handle_audiosocket_conn(_BoomReader(), writer)
        assert writer.closed


# ── Bridge failure + call teardown ─────────────────────────────────────────

class TestCallTeardown:

    @pytest.mark.asyncio
    async def test_bridge_creation_failure_ends_call(self):
        adapter = _make_adapter()

        async def fake_ari(method, path, params=None):
            if path == "channels/externalMedia":
                return {"id": "ext-1"}
            if path == "bridges":
                return None  # bridge creation fails
            return None

        adapter._ari_request = AsyncMock(side_effect=fake_ari)
        adapter._end_call = AsyncMock()
        await adapter._on_stasis_start(
            {"channel": {"id": "chan-C", "name": "PJSIP/obi200", "caller": {}}})
        adapter._end_call.assert_awaited_once_with("chan-C", hangup=True)

    @pytest.mark.asyncio
    async def test_end_call_releases_bridge_channel_and_writer(self):
        adapter = _make_adapter()
        call = _Call("chan-1", "555")
        call.bridge_id = "br-1"
        call.writer = _FakeWriter()
        call.playback_task = asyncio.ensure_future(asyncio.sleep(60))
        adapter._calls["chan-1"] = call
        adapter._uuid_to_channel[call.media_uuid] = "chan-1"
        deletes = []
        adapter._ari_request = AsyncMock(
            side_effect=lambda m, p, params=None: deletes.append((m, p)))

        await adapter._end_call("chan-1", hangup=True)

        assert ("DELETE", "bridges/br-1") in deletes
        assert ("DELETE", "channels/chan-1") in deletes
        assert call.writer.closed
        # The in-flight playback task was cancelled; let it settle.
        with pytest.raises(asyncio.CancelledError):
            await call.playback_task
        assert "chan-1" not in adapter._calls
        assert call.media_uuid not in adapter._uuid_to_channel

    @pytest.mark.asyncio
    async def test_end_call_unknown_channel_is_noop(self):
        adapter = _make_adapter()
        adapter._ari_request = AsyncMock()
        await adapter._end_call("ghost", hangup=True)
        adapter._ari_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_ws_failure_tears_down_and_returns_false(self):
        adapter = _make_adapter(ari_url="http://pbx:8088", ari_user="u", ari_password="p")
        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()
        fake_session = MagicMock()
        fake_session.ws_connect = AsyncMock(side_effect=RuntimeError("ws refused"))
        fake_session.close = AsyncMock()

        with patch.object(_sip.asyncio, "start_server",
                          new=AsyncMock(return_value=fake_server)), \
             patch("aiohttp.ClientSession", return_value=fake_session), \
             patch("aiohttp.BasicAuth", return_value=MagicMock()):
            assert await adapter.connect() is False
        assert not adapter.is_connected
        fake_session.close.assert_awaited()   # transports torn down on failure


# ── A few more real-logic branches ─────────────────────────────────────────

class TestMoreRealBranches:

    def test_attach_media_accepts_ascii_uuid_payload(self):
        # Some Asterisk builds send the UUID as an ASCII string, not 16 bytes.
        adapter = _make_adapter()
        call = _Call("chan-1", "555")
        adapter._calls["chan-1"] = call
        adapter._uuid_to_channel[call.media_uuid] = "chan-1"
        writer = _FakeWriter()
        attached = adapter._attach_media(call.media_uuid.encode("ascii"), writer)
        assert attached is call
        assert call.writer is writer

    @pytest.mark.asyncio
    async def test_synthesize_slin_empty_tts_file_returns_empty(self):
        adapter = _make_adapter()

        def fake_tts(text, path):
            # Report success but write nothing (zero-byte file).
            open(path, "wb").close()
            return '{"success": true}'

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
            assert await adapter._synthesize_slin("hi") == b""

    @pytest.mark.asyncio
    async def test_stream_frames_stops_when_socket_closes(self):
        adapter = _make_adapter()
        call = _Call("chan-1", "555")
        call.writer = _FakeWriter()
        call.writer.closed = True  # socket already closing → no frames written
        with patch.object(_sip.asyncio, "sleep", new=AsyncMock()):
            await adapter._stream_frames(call, b"\x01" * (SLIN_FRAME_BYTES * 2))
        assert bytes(call.writer.buf) == b""
