#!/usr/bin/env python3
"""Exercise the packaged Rust ABI without opening the app or the user's library."""
import ctypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request

if not os.environ.get("PADDOCK_DATA") or not os.environ.get("XDG_RUNTIME_DIR"):
    raise SystemExit("An isolated data/runtime directory is required.")

library = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
pointer = ctypes.c_void_p
library.paddock_desktop_abi_version.restype = ctypes.c_uint32
library.paddock_desktop_open.argtypes = [ctypes.POINTER(pointer)]
library.paddock_desktop_open.restype = pointer
library.paddock_desktop_snapshot.argtypes = [pointer, ctypes.POINTER(pointer)]
library.paddock_desktop_snapshot.restype = pointer
library.paddock_desktop_studio.argtypes = [pointer, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(pointer)]
library.paddock_desktop_studio.restype = pointer
library.paddock_desktop_maintenance.argtypes = [pointer, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(pointer)]
library.paddock_desktop_maintenance.restype = pointer
library.paddock_desktop_string_free.argtypes = [pointer]
library.paddock_desktop_string_free.restype = None
library.paddock_desktop_close.argtypes = [pointer]
library.paddock_desktop_close.restype = None
assert library.paddock_desktop_abi_version() == 12, "Update the smoke contract for the new ABI."


def checked(value, error):
    if error.value:
        message = ctypes.string_at(error).decode()
        library.paddock_desktop_string_free(error)
        raise RuntimeError(message)
    if not value:
        raise RuntimeError("Empty FFI result")
    return value


def maintenance(core, command):
    raw = json.dumps(command).encode()
    error = pointer()
    data = checked(library.paddock_desktop_maintenance(core, raw, len(raw), ctypes.byref(error)), error)
    try:
        return json.loads(ctypes.string_at(data))
    finally:
        library.paddock_desktop_string_free(data)


def inspect(core, command):
    receipt = maintenance(core, command)
    ticket = receipt["id"]
    try:
        deadline = time.monotonic() + 30
        while receipt["state"] == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
            receipt = maintenance(core, {"kind": "poll", "id": ticket})
        assert receipt["state"] == "complete", f"Management command failed: {receipt.get('message', 'timeout')}"
        return json.loads(receipt["payload"])
    finally:
        maintenance(core, {"kind": "close", "id": ticket})


class ReadingFixture(BaseHTTPRequestHandler):
    """An ephemeral runner fixture, never an installed or user-configured model."""
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        valid = (self.path == "/v1/systemone" and body.get("model") == "smoke-reader"
                 and body.get("state") == "The sky is blue.")
        self.send_response(200 if valid else 400)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"answers": {"q1": {"type": "noul", "noul": 0.9}}}).encode())


def reads(core):
    error = pointer()
    data = checked(library.paddock_desktop_studio(core, b"{}", 2, ctypes.byref(error)), error)
    try:
        host = json.loads(ctypes.string_at(data))
    finally:
        library.paddock_desktop_string_free(data)

    def call(path, method="GET", body=None, authenticate=True):
        headers = {"Content-Type": "application/json"}
        if authenticate:
            headers["Cookie"] = host["cookieName"] + "=" + host["session"]
        request = urllib.request.Request(host["origin"] + path, method=method, headers=headers,
                                         data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
            return json.loads(raw) if raw else None

    try:
        call("/api/reads", authenticate=False)
        raise AssertionError("Reads allowed access without the native session")
    except urllib.error.HTTPError as failure:
        assert failure.code == 401
    assert call("/api/reads") == []
    try:
        call("/api/read-runs/draft", authenticate=False)
        raise AssertionError("Read history allowed access without the native session")
    except urllib.error.HTTPError as failure:
        assert failure.code == 401
    if cycle == 0:
        assert call("/api/read-runs/draft") == []
        call("/api/read-runs/draft", "POST", {
            "id": str(uuid.uuid4()), "at": 1, "fingerprint": "a" * 64,
            "excerpt": "Short excerpt", "characters": 9999, "questions": [{}],
            "raw": {"answers": {}}, "port": 1234, "elapsedMilliseconds": 10,
        })
    else:
        history = call("/api/read-runs/draft")
        assert len(history) == 1 and history[0]["excerpt"] == "Short excerpt"
        assert "state" not in history[0] and "request" not in history[0]
        call("/api/read-runs/draft", "DELETE")
        assert call("/api/read-runs/draft") == []
    saved = call("/api/reads", "POST", {"id": "smoke-set", "name": "Check",
                 "body": "{}", "revision": ""})["set"]
    assert call("/api/reads/smoke-set")["revision"] == saved["revision"]
    call("/api/reads/smoke-set?revision=" + saved["revision"], "DELETE")
    assert call("/api/reads") == []
    fixture = ThreadingHTTPServer(("127.0.0.1", 0), ReadingFixture)
    thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    thread.start()
    try:
        result = call(f"/api/runners/{fixture.server_port}/v1/systemone", "POST", {
            "model": "smoke-reader", "state": "The sky is blue.", "samples": 1,
            "questions": {"q1": {"type": "noul", "instructions": "Is the sky blue?"}}})
        assert result["answers"]["q1"]["noul"] == 0.9
    finally:
        fixture.shutdown()
        fixture.server_close()
        thread.join(timeout=5)


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
        assert inspect(core, {"kind": "cache"})["servers"] == []
        assert inspect(core, {"kind": "usage", "from": 0, "to": 10000})["buckets"] == []
        assert inspect(core, {"kind": "activity"})["events"] == []
        assert inspect(core, {"kind": "profiles", "model": "fixture", "artifact": "fixture"})["profiles"] == []
        assert inspect(core, {"kind": "benchmark_history"})["reports"] == []
        reads(core)
    finally:
        library.paddock_desktop_close(core)
print("Packaged desktop ABI: open/snapshot/management/Reads relay/close/reopen passed (isolated data; fixture runner).")
