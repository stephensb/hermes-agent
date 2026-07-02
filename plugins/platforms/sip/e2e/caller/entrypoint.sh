#!/bin/sh
# Synthesize a spoken phrase, pack it as PCMU RTP, then drive one SIP call
# (REGISTER + INVITE + play audio + BYE) into Asterisk with sipp.
set -eu

ASTERISK_HOST="${ASTERISK_HOST:-asterisk}"
SIP_USER="${SIP_USER:-obi200}"
SIP_PASSWORD="${SIP_PASSWORD:-change-me}"
DIAL_EXT="${DIAL_EXT:-100}"
SAY="${SAY:-hello hermes, what time is it}"
CALL_SECONDS="${CALL_SECONDS:-8}"

echo "caller: synthesizing \"$SAY\""
espeak-ng -w /tmp/say.wav "$SAY"
# 8 kHz mono raw µ-law — the exact codec Asterisk offers the OBi200.
ffmpeg -hide_banner -loglevel error -y -i /tmp/say.wav \
    -ar 8000 -ac 1 -f mulaw /tmp/say.ulaw
python3 make_pcap.py /tmp/say.ulaw /tmp/say.pcap
echo "caller: built $(wc -c < /tmp/say.pcap) byte pcap"

# CALL_PAUSE_MS: hold the call up after playback so Hermes can transcribe,
# think, and speak its reply back down the bridge.
CALL_PAUSE_MS=$((CALL_SECONDS * 1000))

echo "caller: dialing ${DIAL_EXT}@${ASTERISK_HOST} as ${SIP_USER}"
exec sipp "${ASTERISK_HOST}:5060" \
    -sf uac.xml \
    -s "${DIAL_EXT}" \
    -au "${SIP_USER}" -ap "${SIP_PASSWORD}" \
    -key user "${SIP_USER}" \
    -key pause_ms "${CALL_PAUSE_MS}" \
    -m 1 -l 1 -r 1 \
    -nostdin -timeout 60s -trace_err
