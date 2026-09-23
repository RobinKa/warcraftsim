"""MPQ archive access through StormLib (ctypes).

Used for reading the game archives (War3*.mpq) and for reading/rewriting map
archives (.w3x/.w3m are MPQs with a 512-byte map header in front).
"""

from __future__ import annotations

import ctypes
import fnmatch
import shutil
from ctypes import POINTER, byref, c_bool, c_char, c_char_p, c_uint, c_ulonglong, c_void_p
from pathlib import Path
from typing import Iterable, Iterator

from .. import paths

MAX_PATH = 1024
MPQ_OPEN_READ_ONLY = 0x00000100
SFILE_OPEN_FROM_MPQ = 0
MPQ_FILE_COMPRESS = 0x00000200
MPQ_FILE_REPLACEEXISTING = 0x80000000
MPQ_COMPRESSION_ZLIB = 0x02
ERROR_NO_MORE_FILES = 18


class _FindData(ctypes.Structure):
    _fields_ = [
        ("cFileName", c_char * MAX_PATH),
        ("szPlainName", c_char_p),
        ("dwHashIndex", c_uint),
        ("dwBlockIndex", c_uint),
        ("dwFileSize", c_uint),
        ("dwFileFlags", c_uint),
        ("dwCompSize", c_uint),
        ("dwFileTimeLo", c_uint),
        ("dwFileTimeHi", c_uint),
        ("lcLocale", c_uint),
    ]


_lib: ctypes.CDLL | None = None


def _storm() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    if not paths.STORMLIB_PATH.exists():
        raise FileNotFoundError(
            f"StormLib not built at {paths.STORMLIB_PATH}; run scripts/build_native.sh"
        )
    lib = ctypes.CDLL(str(paths.STORMLIB_PATH))
    sigs = {
        "SFileOpenArchive": (c_bool, [c_char_p, c_uint, c_uint, POINTER(c_void_p)]),
        "SFileCloseArchive": (c_bool, [c_void_p]),
        "SFileFlushArchive": (c_bool, [c_void_p]),
        "SFileCompactArchive": (c_bool, [c_void_p, c_char_p, c_bool]),
        "SFileHasFile": (c_bool, [c_void_p, c_char_p]),
        "SFileOpenFileEx": (c_bool, [c_void_p, c_char_p, c_uint, POINTER(c_void_p)]),
        "SFileGetFileSize": (c_uint, [c_void_p, POINTER(c_uint)]),
        "SFileReadFile": (c_bool, [c_void_p, c_void_p, c_uint, POINTER(c_uint), c_void_p]),
        "SFileCloseFile": (c_bool, [c_void_p]),
        "SFileFindFirstFile": (c_void_p, [c_void_p, c_char_p, POINTER(_FindData), c_char_p]),
        "SFileFindNextFile": (c_bool, [c_void_p, POINTER(_FindData)]),
        "SFileFindClose": (c_bool, [c_void_p]),
        "SFileCreateFile": (
            c_bool,
            [c_void_p, c_char_p, c_ulonglong, c_uint, c_uint, c_uint, POINTER(c_void_p)],
        ),
        "SFileWriteFile": (c_bool, [c_void_p, c_void_p, c_uint, c_uint]),
        "SFileFinishFile": (c_bool, [c_void_p]),
        "SFileRemoveFile": (c_bool, [c_void_p, c_char_p, c_uint]),
        "SFileSetMaxFileCount": (c_bool, [c_void_p, c_uint]),
        "SErrGetLastError": (c_uint, []),
    }
    for name, (restype, argtypes) in sigs.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    _lib = lib
    return lib


class MpqError(OSError):
    pass


def _fail(what: str) -> MpqError:
    code = _storm().SErrGetLastError()
    return MpqError(code, f"{what} failed (StormLib error {code})")


def _name(name: str) -> bytes:
    # MPQ names use backslashes; the game data is ASCII.
    return name.replace("/", "\\").encode("latin-1")


class MpqArchive:
    """An open MPQ archive. Use as a context manager."""

    def __init__(self, path: str | Path, writable: bool = False):
        self.path = Path(path)
        self.writable = writable
        handle = c_void_p()
        flags = 0 if writable else MPQ_OPEN_READ_ONLY
        if not _storm().SFileOpenArchive(str(self.path).encode(), 0, flags, byref(handle)):
            raise _fail(f"open {self.path}")
        self._h = handle

    def close(self) -> None:
        if self._h:
            _storm().SFileCloseArchive(self._h)
            self._h = c_void_p()

    def __enter__(self) -> "MpqArchive":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __contains__(self, name: str) -> bool:
        return bool(_storm().SFileHasFile(self._h, _name(name)))

    def read(self, name: str) -> bytes:
        lib = _storm()
        fh = c_void_p()
        if not lib.SFileOpenFileEx(self._h, _name(name), SFILE_OPEN_FROM_MPQ, byref(fh)):
            raise _fail(f"open {name} in {self.path.name}")
        try:
            size = lib.SFileGetFileSize(fh, None)
            buf = ctypes.create_string_buffer(size)
            got = c_uint(0)
            if not lib.SFileReadFile(fh, buf, size, byref(got), None) and got.value != size:
                raise _fail(f"read {name}")
            return buf.raw[: got.value]
        finally:
            lib.SFileCloseFile(fh)

    def names(self, mask: str = "*") -> list[str]:
        """File names known to the archive's (listfile), filtered by a glob mask."""
        lib = _storm()
        data = _FindData()
        found: list[str] = []
        hfind = lib.SFileFindFirstFile(self._h, b"*", byref(data), None)
        if not hfind:
            return found
        try:
            while True:
                found.append(data.cFileName.decode("latin-1"))
                if not lib.SFileFindNextFile(hfind, byref(data)):
                    break
        finally:
            lib.SFileFindClose(hfind)
        if mask != "*":
            found = [n for n in found if fnmatch.fnmatch(n.lower(), mask.lower())]
        return found

    def write(self, name: str, data: bytes) -> None:
        """Add or replace a file (zlib-compressed)."""
        if not self.writable:
            raise MpqError(0, "archive opened read-only")
        lib = _storm()
        fh = c_void_p()
        flags = MPQ_FILE_COMPRESS | MPQ_FILE_REPLACEEXISTING
        if not lib.SFileCreateFile(self._h, _name(name), 0, len(data), 0, flags, byref(fh)):
            code = lib.SErrGetLastError()
            if code != 105:  # ERROR_DISK_FULL: hash table full -> grow and retry
                raise _fail(f"create {name}")
            count = len(self.names()) * 2 + 16
            if not lib.SFileSetMaxFileCount(self._h, count):
                raise _fail("grow hash table")
            if not lib.SFileCreateFile(self._h, _name(name), 0, len(data), 0, flags, byref(fh)):
                raise _fail(f"create {name}")
        if data and not lib.SFileWriteFile(fh, data, len(data), MPQ_COMPRESSION_ZLIB):
            lib.SFileFinishFile(fh)
            raise _fail(f"write {name}")
        if not lib.SFileFinishFile(fh):
            raise _fail(f"finish {name}")

    def remove(self, name: str) -> None:
        if not _storm().SFileRemoveFile(self._h, _name(name), SFILE_OPEN_FROM_MPQ):
            raise _fail(f"remove {name}")

    def flush(self) -> None:
        if not _storm().SFileFlushArchive(self._h):
            raise _fail("flush")


class GameArchives:
    """Layered read access to the game's base archives (first archive wins)."""

    def __init__(self, game_dir: str | Path | None = None, archives: Iterable[str] = paths.GAME_ARCHIVES):
        self.game_dir = Path(game_dir) if game_dir else paths.GAME_DIR
        self._archives = [
            MpqArchive(self.game_dir / a) for a in archives if (self.game_dir / a).exists()
        ]
        if not self._archives:
            raise FileNotFoundError(f"no game archives found in {self.game_dir}")

    def close(self) -> None:
        for a in self._archives:
            a.close()

    def __enter__(self) -> "GameArchives":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __contains__(self, name: str) -> bool:
        return any(name in a for a in self._archives)

    def read(self, name: str) -> bytes:
        for a in self._archives:
            if name in a:
                return a.read(name)
        raise FileNotFoundError(name)

    def source_of(self, name: str) -> str | None:
        for a in self._archives:
            if name in a:
                return a.path.name
        return None

    def names(self, mask: str = "*") -> Iterator[str]:
        seen: set[str] = set()
        for a in self._archives:
            for n in a.names(mask):
                key = n.lower()
                if key not in seen:
                    seen.add(key)
                    yield n


def copy_and_edit(src: str | Path, dst: str | Path) -> MpqArchive:
    """Copy a map archive to dst and open the copy for writing."""
    shutil.copyfile(src, dst)
    return MpqArchive(dst, writable=True)
