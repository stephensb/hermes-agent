#!/usr/bin/env python3
"""Pack raw 8 kHz G.711 µ-law audio into an Ethernet/IP/UDP/RTP pcap that
sipp can replay with ``exec play_pcap_audio``.

sipp ignores the pcap's own IP/ports (it rewrites them to the negotiated
media address), so only the RTP payloads and inter-packet timing matter.
We emit one PCMU packet per 20 ms (160 µ-law bytes, payload type 0).

Usage:
    make_pcap.py <input.ulaw> <output.pcap>
"""
import struct
import sys


def ip_checksum(header: bytes) -> int:
    s = 0
    for i in range(0, len(header), 2):
        s += (header[i] << 8) + header[i + 1]
    s = (s >> 16) + (s & 0xFFFF)
    s += s >> 16
    return (~s) & 0xFFFF


def build(ulaw: bytes) -> bytes:
    SAMPLES = 160          # 20 ms @ 8 kHz
    PT_PCMU = 0
    ssrc = 0x0BADCAFE
    src_mac = bytes.fromhex("020000000001")
    dst_mac = bytes.fromhex("020000000002")
    src_ip = bytes([10, 0, 0, 1])
    dst_ip = bytes([10, 0, 0, 2])
    src_port, dst_port = 40000, 40002

    # pcap global header: little-endian magic, v2.4, Ethernet (LINKTYPE 1).
    out = bytearray()
    out += struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)

    seq = 0
    ts_rtp = 0
    usec = 0
    for off in range(0, len(ulaw), SAMPLES):
        payload = ulaw[off:off + SAMPLES]
        if len(payload) < SAMPLES:
            payload = payload + b"\xff" * (SAMPLES - len(payload))  # µ-law silence
        marker = 0x80 if seq == 0 else 0x00
        rtp = struct.pack(">BBHII", 0x80, marker | PT_PCMU, seq & 0xFFFF,
                          ts_rtp & 0xFFFFFFFF, ssrc) + payload

        udp_len = 8 + len(rtp)
        udp = struct.pack(">HHHH", src_port, dst_port, udp_len, 0) + rtp

        total_len = 20 + udp_len
        ip_no_csum = struct.pack(">BBHHHBBH", 0x45, 0, total_len, seq & 0xFFFF,
                                 0x4000, 64, 17, 0) + src_ip + dst_ip
        csum = ip_checksum(ip_no_csum)
        ip = ip_no_csum[:10] + struct.pack(">H", csum) + ip_no_csum[12:]

        frame = dst_mac + src_mac + b"\x08\x00" + ip + udp

        ts_sec, ts_frac = divmod(usec, 1_000_000)
        out += struct.pack("<IIII", ts_sec, ts_frac, len(frame), len(frame))
        out += frame

        seq += 1
        ts_rtp += SAMPLES
        usec += 20_000  # 20 ms between packets

    return bytes(out)


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: make_pcap.py <input.ulaw> <output.pcap>\n")
        return 2
    with open(sys.argv[1], "rb") as f:
        ulaw = f.read()
    with open(sys.argv[2], "wb") as f:
        f.write(build(ulaw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
