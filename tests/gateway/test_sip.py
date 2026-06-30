"""Tests for the SIP voice-bridge platform adapter plugin.

These cover the pure, hardware-free pieces: AudioSocket framing, the VAD
turn detector, PCM framing, config/env parsing, and plugin registration.
The live ARI + AudioSocket round-trip needs a real Asterisk and is verified
manually (see plugins/platforms/sip/asterisk/README.md).
"""

import struct

import pytest
from unittest.mock import MagicMock

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
SLIN_FRAME_BYTES = _sip.SLIN_FRAME_BYTES


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
