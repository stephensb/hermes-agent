# Deterministic Asterisk for the SIP e2e harness.
#
# Debian's `asterisk` package ships chan_pjsip, res_ari, res_http_websocket
# and app_audiosocket (the modules the Hermes bridge needs), and autoloads
# modules by default — so the mounted pjsip/extensions/ari/http .conf files
# are all that's required.
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends asterisk curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run in the foreground as PID 1 with console logging so `docker compose logs`
# shows Stasis / AudioSocket activity.  -p enables realtime priority best-effort.
EXPOSE 5060/udp 8088/tcp 10000-10100/udp
CMD ["asterisk", "-f", "-vvvvv"]
