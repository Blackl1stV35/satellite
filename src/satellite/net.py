"""Polite network layer: one request per interval per host, range reads for rasterio, RAM gate."""

from __future__ import annotations

import threading
import time
from urllib.parse import urlsplit

import requests

MIN_INTERVAL_S = 2.0
USER_AGENT = "satellite-research/0.1 (read-only GET, descriptive research)"
_lock = threading.Lock()
_last: dict[str, float] = {}
_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
MAX_READ = 4 << 20
PROBE_READ = 16 << 20
STATS = {"requests": 0}


def wait_turn(host: str, interval: float | None = None, clock=time.monotonic, sleep=time.sleep) -> float:
    """Block until at least `interval` seconds passed since the previous request to host."""
    interval = MIN_INTERVAL_S if interval is None else interval
    with _lock:
        now = clock()
        gap = _last.get(host, -1e9) + interval - now
        if gap > 0:
            sleep(gap)
        _last[host] = clock()
        return max(gap, 0.0)


def get(url: str, headers: dict | None = None, timeout: float = 60, **kw) -> requests.Response:
    wait_turn(urlsplit(url).netloc)
    STATS["requests"] += 1
    return _session.get(url, headers=headers, timeout=timeout, **kw)


def post_json(url: str, payload: dict, timeout: float = 60) -> requests.Response:
    wait_turn(urlsplit(url).netloc)
    STATS["requests"] += 1
    return _session.post(url, json=payload, timeout=timeout)


class _Shared:
    """Block cache and size for one URL, shared by all handles opened on it."""

    def __init__(self, url: str, block: int, max_blocks: int):
        self.url = url
        self.block = block
        self.max_blocks = max_blocks
        self.cache: dict[int, bytes] = {}
        self.order: list[int] = []
        self.size: int | None = None

    def fetch(self, idx: int) -> bytes:
        if idx in self.cache:
            return self.cache[idx]
        start = idx * self.block
        r = get(self.url, headers={"Range": f"bytes={start}-{start + self.block - 1}"})
        if r.status_code not in (200, 206):
            raise OSError(f"HTTP {r.status_code} for range read")
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            self.size = int(cr.rsplit("/", 1)[1])
        elif self.size is None:
            self.size = len(r.content)
        self.cache[idx] = r.content
        self.order.append(idx)
        if len(self.order) > self.max_blocks:
            self.cache.pop(self.order.pop(0), None)
        return r.content


_SHARED: dict[str, _Shared] = {}


def clear_cache() -> None:
    _SHARED.clear()


def forget(url: str) -> None:
    """Drop the block cache of one URL (call after the file is closed)."""
    _SHARED.pop(url.split("?")[0], None)


class RangeFile:
    """Read-only seekable file over HTTP range requests with aligned block caching."""

    def __init__(self, url: str, block: int = 2 << 20, max_blocks: int = 16):
        key = url.split("?")[0]
        sh = _SHARED.get(key)
        if sh is None:
            sh = _SHARED[key] = _Shared(url, block, max_blocks)
        self.sh = sh
        self.pos = 0
        if sh.size is None:
            sh.fetch(0)

    @property
    def size(self):
        return self.sh.size

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = (self.sh.size or 0) + off
        return self.pos

    def read(self, n=-1):
        size = self.sh.size or 0
        if n is None or n < 0:
            n = size - self.pos
        if n > PROBE_READ:  # GDAL probes with a whole-file read; answer with a short read
            n = 16384
        end = min(self.pos + n, size, self.pos + MAX_READ)
        out = bytearray()
        p = self.pos
        blk = self.sh.block
        while p < end:
            idx = p // blk
            data = self.sh.fetch(idx)
            off = p - idx * blk
            take = min(end - p, len(data) - off)
            if take <= 0:
                break
            out += data[off : off + take]
            p += take
        self.pos = p
        return bytes(out)

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def open_range(path, mode="rb", **kw):
    if not str(path).startswith("http"):
        import io

        return io.BytesIO(b"")
    return RangeFile(path)


def free_ram_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / 2**30


def wait_for_ram(min_gb: float = 1.5, poll_s: float = 30, max_wait_s: float = 6 * 3600, log=print) -> float:
    """Block until at least min_gb of RAM is available; returns the value seen."""
    t0 = time.monotonic()
    while True:
        g = free_ram_gb()
        if g >= min_gb:
            return g
        if time.monotonic() - t0 > max_wait_s:
            raise MemoryError(f"only {g:.2f} GB free after waiting")
        log(f"waiting for RAM: {g:.2f} GB free, need {min_gb}")
        time.sleep(poll_s)
