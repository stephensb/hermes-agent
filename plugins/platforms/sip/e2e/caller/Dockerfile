# Simulated OBi200 for the SIP e2e harness: sipp registers as the ATA and
# places a call carrying a spoken phrase (espeak-ng → ffmpeg µ-law → pcap).
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        sip-tester espeak-ng ffmpeg python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /caller
COPY make_pcap.py uac.xml entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
