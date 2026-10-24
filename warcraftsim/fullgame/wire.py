"""The remote actors' wire format (fullgame/gateway.py on the learner's machine, fullgame/remoteactor.py on
another): frames over TCP, a handshake with a shared key, and the messages' encodings.

* A frame: its payload's length (uint32), its kind and flags (a byte each), the payload (compressed with
  zstd, or zlib where the other side has no zstd: Python before 3.14 without the zstandard package).
* The handshake: the gateway sends a nonce, the other side the nonce's HMAC under the shared key (a file on
  both machines, KEY_PATH) and its hello (JSON); the gateway answers with its own. A wrong key: closed.
* What the learner reads is never pickled: values go as JSON with their arrays' bytes after it (pack_value),
  so a frame from the network can be no more than data.
* A network call (REQUEST): a game's sides' view steps, the entities' features in float16 (as Trajectory
  keeps them for the learner: the policy acts on what it is trained on), only the step's n entities; the
  answer (REPLY): the orders, their log-probabilities, the value and the memory's state, as SharedRows.answer
  gives them. A view step's ~13 KB in float16 is ~2.8 KB with zstd (recorded Echo Isles games, 56 entities).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import socket
import struct
import zlib
from pathlib import Path

import numpy as np

try:
    from compression import zstd as _zstd  # (Python 3.14)
except ImportError:
    try:
        import zstandard as _zstandard  # noqa: F401

        class _zstd:  # (the zstandard package's, as compression.zstd's two calls)
            @staticmethod
            def compress(b: bytes, level: int = 1) -> bytes:
                return _zstandard.ZstdCompressor(level=level).compress(b)

            @staticmethod
            def decompress(b: bytes) -> bytes:
                return _zstandard.ZstdDecompressor().decompress(b)
    except ImportError:
        _zstd = None

MAGIC = b"WSIM1"
KEY_PATH = Path.home() / ".config" / "warcraftsim" / "remote.key"
HEADER = struct.Struct("<IBB")
MAX_FRAME = 1 << 28
# frame kinds
HELLO, REQUEST, REPLY, PUT, SPEC_ASK, SPEC, BUSY, BYE, READY = range(1, 10)
# flags
ZSTD, ZLIB = 1, 2
CODECS = ("zstd", "zlib") if _zstd is not None else ("zlib",)


class WireError(ConnectionError):
    """The other side closed, or sent what this side cannot read."""


def load_key(path: Path | None = None, create: bool = False) -> bytes:
    """The shared key (hex in a file only its user reads); created with `create` when missing."""
    path = Path(path or KEY_PATH)
    if not path.exists():
        if not create:
            raise FileNotFoundError(f"no key at {path}: copy the learner machine's {KEY_PATH} there")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32) + "\n")
    return bytes.fromhex(path.read_text().strip())


# ---- frames ---------------------------------------------------------------------------------------

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view, at = memoryview(buf), 0
    while at < n:
        got = sock.recv_into(view[at:], n - at)
        if not got:
            raise WireError("connection closed")
        at += got
    return bytes(buf)


def compress(data: bytes, codec: str | None) -> tuple[bytes, int]:
    if codec == "zstd" and _zstd is not None:
        return _zstd.compress(data, 1), ZSTD
    if codec == "zlib":
        return zlib.compress(data, 1), ZLIB
    return data, 0


def decompress(data: bytes, flags: int) -> bytes:
    if flags & ZSTD:
        if _zstd is None:
            raise WireError("a zstd frame, and no zstd here")
        return _zstd.decompress(data)
    if flags & ZLIB:
        return zlib.decompress(data)
    return data


def send_frame(sock: socket.socket, kind: int, payload: bytes, codec: str | None = None) -> int:
    """-> the bytes sent."""
    data, flags = compress(payload, codec) if codec and len(payload) > 256 else (payload, 0)
    sock.sendall(HEADER.pack(len(data), kind, flags) + data)
    return HEADER.size + len(data)


def recv_frame(sock: socket.socket) -> tuple[int, bytes, int]:
    """-> (kind, payload, the bytes received)."""
    n, kind, flags = HEADER.unpack(_recv_exact(sock, HEADER.size))
    if n > MAX_FRAME:
        raise WireError(f"a frame of {n} bytes")
    return kind, decompress(_recv_exact(sock, n), flags), HEADER.size + n


# ---- the handshake --------------------------------------------------------------------------------

def _mac(key: bytes, nonce: bytes) -> bytes:
    return hmac.new(key, nonce + b"client", hashlib.sha256).digest()


class Refused(WireError):
    """The gateway will not take this side (its code is not the learner's): trying again will not help."""


_CODE: dict[str, str] = {}


def code_version(root: Path | None = None) -> str:
    """A hash of what the actors' games and features run: the package's Python and JASS that they import (not
    the dashboard's, the scripts'), the native sources and the built shim. A remote machine's must be the
    learner's: other features, other orders or another harness would train the policy on what it does not do
    here."""
    from .. import paths
    root = Path(root or paths.REPO_ROOT)
    if str(root) not in _CODE:
        pkg = root / "warcraftsim"
        files = [p for p in pkg.glob("*.py")] + [p for d in ("fullgame", "runtime", "harness", "data", "rl")
                                                  for p in (pkg / d).rglob("*") if p.suffix in (".py", ".j")
                                                  and "__pycache__" not in p.parts]
        files += [p for p in (root / "native").rglob("*") if p.suffix in (".c", ".h", ".cpp")]
        files += [root / "build" / "shim" / f for f in ("w3shim.dll", "w3launch.exe") if (root / "build" / "shim" / f).exists()]
        h = hashlib.sha256()
        for p in sorted(files):
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
        _CODE[str(root)] = h.hexdigest()
    return _CODE[str(root)]


def server_handshake(sock: socket.socket, key: bytes, hello: dict | None = None, code: str | None = None) -> dict:
    """The gateway's side: -> the client's hello (WireError: not a client with the key; with `code`, the
    learner's code_version, one with other code is told so and refused)."""
    nonce = secrets.token_bytes(32)
    sock.sendall(MAGIC + nonce)
    mac = _recv_exact(sock, 32)
    if not hmac.compare_digest(mac, _mac(key, nonce)):
        raise WireError("wrong key")
    kind, payload, _ = recv_frame(sock)
    if kind != HELLO:
        raise WireError(f"frame {kind} before the hello")
    theirs = json.loads(payload)
    if code is not None and theirs.get("code") != code:
        why = (f"its code ({str(theirs.get('code'))[:12]}) is not the learner's ({code[:12]}): mirror the learner's tree "
               "(warcraftsim/, native/, the built shim) and start again")
        send_frame(sock, HELLO, json.dumps({"refused": why}).encode())
        raise Refused(why)
    mine = dict(hello or {})
    mine["codec"] = next((c for c in theirs.get("codecs", ["zlib"]) if c in CODECS), None)
    send_frame(sock, HELLO, json.dumps(mine).encode())
    return {**theirs, "codec": mine["codec"]}


def client_handshake(sock: socket.socket, key: bytes, hello: dict) -> dict:
    """The remote side's: -> the gateway's hello (its "codec": the frames' compression both sides have).
    Refused: the gateway will not take this machine's code."""
    got = _recv_exact(sock, len(MAGIC) + 32)
    if got[:len(MAGIC)] != MAGIC:
        raise WireError("not a warcraftsim gateway")
    sock.sendall(_mac(key, got[len(MAGIC):]))
    send_frame(sock, HELLO, json.dumps({"code": code_version(), **hello, "codecs": list(CODECS)}).encode())
    kind, payload, _ = recv_frame(sock)
    if kind != HELLO:
        raise WireError(f"frame {kind} instead of the gateway's hello")
    theirs = json.loads(payload)
    if "refused" in theirs:
        raise Refused(theirs["refused"])
    return theirs


# ---- values: JSON and their arrays' bytes (never pickled) -----------------------------------------

_TAGS = ("__a", "__s", "__t", "__d", "__b")


def _enc(x, bufs: list):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, np.ndarray):
        if x.dtype.hasobject:
            raise TypeError("an object array")
        bufs.append(np.ascontiguousarray(x).tobytes())
        return {"__a": [x.dtype.str, list(x.shape), len(bufs) - 1]}
    if isinstance(x, np.generic):
        return {"__s": [x.dtype.str, x.item()]}
    if isinstance(x, dict):
        if all(isinstance(k, str) and not k.startswith("__") for k in x):
            return {k: _enc(v, bufs) for k, v in x.items()}
        return {"__d": [[_enc(k, bufs), _enc(v, bufs)] for k, v in x.items()]}
    if isinstance(x, list):
        return [_enc(v, bufs) for v in x]
    if isinstance(x, tuple):
        return {"__t": [_enc(v, bufs) for v in x]}
    if isinstance(x, (bytes, bytearray)):
        bufs.append(bytes(x))
        return {"__b": len(bufs) - 1}
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"cannot send a {type(x).__name__}")


def _dec(x, bufs: list):
    if isinstance(x, list):
        return [_dec(v, bufs) for v in x]
    if not isinstance(x, dict):
        return x
    if len(x) == 1:
        (tag, v), = x.items()
        if tag == "__a":
            dtype, shape, i = v
            dt = np.dtype(dtype)
            if dt.hasobject:
                raise WireError("an object array")
            return np.frombuffer(bufs[i], dt).reshape(shape).copy()
        if tag == "__s":
            dt = np.dtype(v[0])
            if dt.hasobject:
                raise WireError("an object scalar")
            return dt.type(v[1])
        if tag == "__t":
            return tuple(_dec(e, bufs) for e in v)
        if tag == "__d":
            return {_dec(k, bufs): _dec(e, bufs) for k, e in v}
        if tag == "__b":
            return bufs[v]
    return {k: _dec(v, bufs) for k, v in x.items()}


def pack_value(x) -> bytes:
    """A value of None, bools, numbers, strings, bytes, lists, tuples, dicts and numpy arrays and scalars (a
    Path as its string) -> bytes: [header length][JSON header][the arrays' bytes, each 8-aligned]."""
    bufs: list[bytes] = []
    head = _enc(x, bufs)
    sizes = [len(b) for b in bufs]
    hj = json.dumps({"v": head, "n": sizes}, separators=(",", ":")).encode()
    out = [struct.pack("<I", len(hj)), hj]
    at = 4 + len(hj)
    for b in bufs:
        pad = -at % 8
        out += [b"\0" * pad, b]
        at += pad + len(b)
    return b"".join(out)


def unpack_value(data: bytes):
    (hn,) = struct.unpack_from("<I", data)
    h = json.loads(data[4:4 + hn])
    bufs, at = [], 4 + hn
    for n in h["n"]:
        at += -at % 8
        if at + n > len(data):
            raise WireError("a value cut short")
        bufs.append(data[at:at + n])
        at += n
    return _dec(h["v"], bufs)


# ---- network calls --------------------------------------------------------------------------------

_ITEM = struct.Struct("<HHHHHBf")  # key length, n, n_own, F, G, flags, guide
_SHARED_NOISE, _AVAIL, _H = 1, 2, 4


def pack_request(items: list[tuple[str, dict]]) -> bytes:
    """[(network key, view step)] -> bytes: per item its sizes, then the entities' features (float16, the n
    used), types and current orders (int16), the global vector (float32), the available orders (bits) and
    the memory's state going in (float32)."""
    out = [struct.pack("<B", len(items))]
    for key, st in items:
        n, kb = int(st["n"]), key.encode()
        ent, glob = st["ent"], np.asarray(st["glob"], np.float32)
        avail, h = st.get("avail"), st.get("h")
        flags = (_SHARED_NOISE if st.get("shared_noise") else 0) | (_AVAIL if avail is not None else 0) | \
            (_H if h is not None else 0)
        out.append(_ITEM.pack(len(kb), n, int(st["n_own"]), ent.shape[1], len(glob), flags, float(st.get("guide") or 0.0)))
        out += [kb, np.ascontiguousarray(ent[:n], np.float16).tobytes(), np.ascontiguousarray(st["type"][:n], np.int16).tobytes(),
                np.ascontiguousarray(st["cur"][:n], np.int16).tobytes(), glob.tobytes()]
        if avail is not None:
            a = np.asarray(avail).astype(bool, copy=False)
            out += [struct.pack("<H", len(a)), np.packbits(a).tobytes()]
        if h is not None:
            hh = np.asarray(h, np.float32)
            out += [struct.pack("<H", len(hh)), hh.tobytes()]
    return b"".join(out)


def unpack_request(data: bytes) -> list[tuple[str, dict]]:
    (count,), at, items = struct.unpack_from("<B", data), 1, []

    def take(nbytes: int) -> bytes:
        nonlocal at
        if at + nbytes > len(data):
            raise WireError("a request cut short")
        b = data[at:at + nbytes]
        at += nbytes
        return b
    for _ in range(count):
        kl, n, n_own, F, G, flags, guide = _ITEM.unpack(take(_ITEM.size))
        key = take(kl).decode()
        st = {"n": n, "n_own": n_own,
              "ent": np.frombuffer(take(2 * n * F), np.float16).reshape(n, F),
              "type": np.frombuffer(take(2 * n), np.int16),
              "cur": np.frombuffer(take(2 * n), np.int16),
              "glob": np.frombuffer(take(4 * G), np.float32)}
        if flags & _AVAIL:
            (na,) = struct.unpack("<H", take(2))
            st["avail"] = np.unpackbits(np.frombuffer(take((na + 7) // 8), np.uint8), count=na).astype(bool)
        if flags & _H:
            (nh,) = struct.unpack("<H", take(2))
            st["h"] = np.frombuffer(take(4 * nh), np.float32)
        if flags & _SHARED_NOISE:
            st["shared_noise"] = True
        if guide:
            st["guide"] = guide
        items.append((key, st))
    return items


_ANS = struct.Struct("<iHHff")  # version (-2: no answer), units, memory width, value, entropy


def pack_reply(answers: list[dict | None]) -> bytes:
    """SharedRows.answer's dicts (None: no answer) -> bytes."""
    out = [struct.pack("<B", len(answers))]
    for res in answers:
        if res is None:
            out.append(_ANS.pack(-2, 0, 0, 0.0, 0.0))
            continue
        o, h = len(res["order"]), res.get("h")
        out.append(_ANS.pack(int(res["version"]), o, len(h) if h is not None else 0, res["value"], res["entropy"]))
        out += [np.asarray(res[k], np.int16).tobytes() for k in ("order", "tgt", "bx", "by")]
        out.append(np.asarray(res["logp"], np.float32).tobytes())
        if h is not None:
            out.append(np.asarray(h, np.float32).tobytes())
    return b"".join(out)


def unpack_reply(data: bytes) -> list[dict | None]:
    (count,), at, out = struct.unpack_from("<B", data), 1, []
    for _ in range(count):
        version, o, m, value, entropy = _ANS.unpack_from(data, at)
        at += _ANS.size
        if version == -2:
            out.append(None)
            continue
        res = {}
        for k in ("order", "tgt", "bx", "by"):
            res[k] = np.frombuffer(data, np.int16, o, at).astype(np.int64)
            at += 2 * o
        res["logp"] = np.frombuffer(data, np.float32, o, at).copy()
        at += 4 * o
        res.update(value=float(np.float32(value)), entropy=float(np.float32(entropy)), version=version)
        if m:
            res["h"] = np.frombuffer(data, np.float32, m, at).copy()
            at += 4 * m
        out.append(res)
    return out
