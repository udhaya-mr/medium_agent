"""Last-known-good DNS fallback.

The Azure endpoint resolves fine almost all the time, but a home or office
router's DNS forwarder drops queries every so often. When that happens
`socket.getaddrinfo` raises gaierror, httpx turns it into ConnectError, and the
openai SDK reports `APIConnectionError: Connection error.` - a network hiccup
several layers away is what kills the agent turn.

This module patches `socket.getaddrinfo` so that a *failed* lookup falls back to
the last address that host resolved to. Live resolution is always tried first,
so this never pins a stale address while DNS is healthy: it only supplies an
answer in the window where the resolver would otherwise return nothing.

Answers are also written to disk, so a cold start during a DNS outage still has
something to work with. Entries expire so a genuinely moved endpoint cannot be
served from cache forever.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dns_cache.json")

# Long enough to ride out an outage and a restart, short enough that an endpoint
# which genuinely moves is not served from cache indefinitely.
_MAX_AGE_SECONDS = 7 * 24 * 60 * 60

_lock = threading.Lock()
_cache: dict[str, dict] = {}
_installed = False


def _load() -> None:
    global _cache
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            _cache = data
    except (OSError, json.JSONDecodeError):
        _cache = {}


def _save() -> None:
    """Best effort - a cache we cannot persist is not worth failing a request over."""
    try:
        tmp = _CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_cache, fh)
        os.replace(tmp, _CACHE_PATH)
    except OSError:
        pass


def _key(host: str, port) -> str:
    return f"{host}:{port}"


def install() -> None:
    """Patch socket.getaddrinfo once. Safe to call repeatedly."""
    global _installed
    with _lock:
        if _installed:
            return
        _installed = True
        _load()

    real_getaddrinfo = socket.getaddrinfo

    def resilient_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        try:
            result = real_getaddrinfo(host, port, family, type, proto, flags)
        except socket.gaierror:
            entry = None
            with _lock:
                entry = _cache.get(_key(host, port))
            if not entry:
                raise
            if time.time() - entry.get("at", 0) > _MAX_AGE_SECONDS:
                raise

            addrs = entry.get("addrs") or []
            if not addrs:
                raise

            print(
                f"[dns_cache] resolver failed for {host}, using last known good "
                f"address {addrs[0][-1][0]}",
                flush=True,
            )
            # getaddrinfo returns tuples; JSON round-trips them as lists, and
            # socket.create_connection unpacks the sockaddr, so rebuild tuples.
            return [(f, t, p, c, tuple(sa)) for f, t, p, c, sa in addrs]

        # Only cache real, routable answers - never a numeric host or empty reply.
        if result and not _is_numeric(host):
            with _lock:
                _cache[_key(host, port)] = {
                    "at": time.time(),
                    "addrs": [[f, t, p, c, list(sa)] for f, t, p, c, sa in result],
                }
                _save()
        return result

    socket.getaddrinfo = resilient_getaddrinfo


def _is_numeric(host) -> bool:
    """True if `host` is already an IP literal, so there is nothing to cache."""
    if not isinstance(host, str):
        return True
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.inet_pton(family, host)
            return True
        except (OSError, ValueError):
            continue
    return False
