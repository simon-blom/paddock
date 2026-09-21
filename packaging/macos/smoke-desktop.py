#!/usr/bin/env python3
"""Exercise the packaged Rust ABI without opening the app or the user's library."""
import ctypes
import json
import os
from pathlib import Path
import sys

if not os.environ.get("PADDOCK_DATA") or not os.environ.get("XDG_RUNTIME_DIR"):
    raise SystemExit("An isolated data/runtime directory is required.")

library = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
pointer = ctypes.c_void_p
library.paddock_desktop_abi_version.restype = ctypes.c_uint32
library.paddock_desktop_open.argtypes = [ctypes.POINTER(pointer)]
library.paddock_desktop_open.restype = pointer
library.paddock_desktop_snapshot.argtypes = [pointer, ctypes.POINTER(pointer)]
library.paddock_desktop_snapshot.restype = pointer
library.paddock_desktop_string_free.argtypes = [pointer]
library.paddock_desktop_string_free.restype = None
library.paddock_desktop_close.argtypes = [pointer]
library.paddock_desktop_close.restype = None
assert library.paddock_desktop_abi_version() == 11, "Update the smoke contract for the new ABI."


def checked(value, error):
    if error.value:
        message = ctypes.string_at(error).decode()
        library.paddock_desktop_string_free(error)
        raise RuntimeError(message)
    if not value:
        raise RuntimeError("Empty FFI result")
    return value


for cycle in range(2):
    error = pointer()
    core = checked(library.paddock_desktop_open(ctypes.byref(error)), error)
    try:
        data = checked(library.paddock_desktop_snapshot(core, ctypes.byref(error)), error)
        try:
            snapshot = json.loads(ctypes.string_at(data))
            assert isinstance(snapshot, dict) and snapshot, "Empty desktop snapshot"
            assert snapshot["identity"]["role"] == "manager", "Wrong embedded core role"
            assert snapshot["catalog"]["schema"] == 3, "Wrong catalog schema"
            assert len(snapshot["catalog"]["models"]) > 10, "Packaged model catalog missing"
            assert snapshot["readiness"]["backend"] == "metal", "Native app selected the wrong backend"
            assert snapshot["servers"] == [], "Isolated smoke test unexpectedly found configured servers"
        finally:
            library.paddock_desktop_string_free(data)
    finally:
        library.paddock_desktop_close(core)
print("Packaged desktop ABI: open/snapshot/close/reopen passed (isolated data).")
