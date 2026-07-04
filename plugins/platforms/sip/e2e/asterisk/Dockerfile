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

# Fail the BUILD (not a mysterious runtime silence) if this Debian release's
# asterisk package ever drops one of the modules the bridge depends on —
# e.g. a rebuild without pjproject, or a future package split. Catching this
# here means a `docker compose build` failure with an actionable module name,
# instead of "no such command 'Stasis'" buried in a dialplan log at call time.
RUN set -e; \
    missing=""; \
    for mod in res_pjsip.so chan_pjsip.so res_ari.so res_ari_channels.so \
               res_http_websocket.so res_stasis.so app_stasis.so \
               app_audiosocket.so chan_audiosocket.so; do \
      if [ ! -f "/usr/lib/asterisk/modules/$mod" ]; then \
        missing="$missing $mod"; \
      fi; \
    done; \
    if [ -n "$missing" ]; then \
      echo "FATAL: required Asterisk module(s) not found in this build:$missing" >&2; \
      echo "The 'asterisk' apt package on this base image apparently doesn't" >&2; \
      echo "ship them. Options: install a package that provides them, or" >&2; \
      echo "build Asterisk from source with pjproject + AudioSocket support." >&2; \
      exit 1; \
    fi

# Run in the foreground as PID 1 with console logging so `docker compose logs`
# shows Stasis / AudioSocket activity.  -p enables realtime priority best-effort.
EXPOSE 5060/udp 8088/tcp 10000-10100/udp
CMD ["asterisk", "-f", "-vvvvv"]
