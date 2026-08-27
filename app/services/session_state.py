"""Optional Redis-backed KV for session/task state, with an in-process fallback.

Nothing in the request path *requires* Redis; when REDIS_URL is unset (or the
server is unreachable at startup) we transparently use a bounded in-memory dict.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


class _MemoryBackend:
    def __init__(self, max_items: int = 512) -> None:
        self._data: OrderedDict[str, str] = OrderedDict()
        self._max = max_items
        self._lock = threading.Lock()

    def set(self, key: str, value: str, ttl: int | None = None) -> None:  # ttl ignored
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._data.get(key)


class SessionStore:
    def __init__(self, redis_url: str | None) -> None:
        self._backend = self._connect(redis_url)

    @property
    def backend_name(self) -> str:
        return "redis" if not isinstance(self._backend, _MemoryBackend) else "memory"

    def _connect(self, redis_url: str | None):
        if not redis_url:
            return _MemoryBackend()
        try:
            import redis

            client = redis.Redis.from_url(
                redis_url, socket_connect_timeout=2, decode_responses=True
            )
            client.ping()
            logger.info("session state: connected to Redis")
            return client
        except Exception as exc:  # noqa: BLE001 - degrade to memory, log why
            logger.warning("session state: Redis unavailable (%s); using in-memory store", exc)
            return _MemoryBackend()

    def put(self, session_id: str, payload: dict, ttl: int = 86400) -> None:
        self._backend.set(f"session:{session_id}", json.dumps(payload, default=str), ttl)

    def fetch(self, session_id: str) -> dict | None:
        raw = self._backend.get(f"session:{session_id}")
        return json.loads(raw) if raw else None
