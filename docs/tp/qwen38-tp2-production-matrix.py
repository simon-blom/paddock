#!/usr/bin/env python3
"""Run fresh-server Qwen3.8 TP2 benchmark samples and preserve raw evidence."""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(os.environ["QWEN38_BENCH_ROOT"])
REPO = Path("/home/sime/repos/paddock")
RUNNER = REPO / "target/release/paddock-runner"
REMOTE_RUNNER = Path("/home/sime/ffn-tp/benchmark-20260928-083340/paddock-runner")
MODEL = Path("/home/sime/models/Qwen3.8-27B-UD-Q4_K_M.gguf")
PACK = REPO / "packs/cuda/build/pd-cuda-sm120.so"
WORKER_MODEL = Path("/home/sime/ffn-tp/Qwen3.8-27B-UD-Q4_K_M.gguf")
WORKER_PACK = Path("/home/sime/ffn-tp/acceptance-f98e/pd-cuda-sm120.so")
LAUNCHER = REPO / "docs/tp/qwen38-tp2-two-node.sh"
CLIENT = REPO / "docs/tp/qwen38-tp2-production-benchmark.py"
PROMPTS = {
    "long": ROOT / "long-prompt.txt",
    "medium": ROOT / "medium-prompt.txt",
}
CONFIGS = {
    "A": {"graph": "off", "spec": "off", "spec_graph": "off"},
    "B": {"graph": "on", "spec": "off", "spec_graph": "off"},
    "C": {"graph": "off", "spec": "on", "spec_graph": "off"},
    "D": {"graph": "on", "spec": "on", "spec_graph": "on"},
}
WORKLOADS = [
    ("long-c1", "long", 1),
    ("medium-c1", "medium", 1),
    ("medium-c2", "medium", 2),
]


def runner_pid(port):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if not argv or not argv[0].decode(errors="replace").endswith("/paddock-runner"):
            continue
        if b"--tp-master-port" in argv:
            i = argv.index(b"--tp-master-port")
            if i + 1 < len(argv) and argv[i + 1] == str(port).encode():
                return int(entry.name)
    return None


def wait_ready(port, proc, log_path, timeout=600):
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"launcher exited {proc.returncode} before readiness; see {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = repr(exc)
        time.sleep(1)
    raise TimeoutError(f"HTTP readiness timeout on {port}: {last_error}; see {log_path}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    run_index = 0
    for config, settings in CONFIGS.items():
        for workload, prompt_name, concurrency in WORKLOADS:
            for rep in range(1, 4):
                run_index += 1
                port = 11600 + run_index
                master_port = 11700 + run_index
                label = f"{config}-{workload}-r{rep}"
                only = os.environ.get("QWEN38_ONLY")
                if only and label != only:
                    continue
                raw_dir = ROOT / "raw" / config / workload / f"r{rep}"
                raw_dir.mkdir(parents=True, exist_ok=True)
                server_log = raw_dir / "coordinator.log"
                remote_log = f"/tmp/paddock-qwen38-{label}-worker.log"
                remote_pidfile = f"/tmp/paddock-qwen38-{label}-worker.pid"
                remote_statusfile = f"/tmp/paddock-qwen38-{label}-worker.status"
                env = os.environ.copy()
                env.update({
                    "RUNNER": str(RUNNER),
                    "REMOTE_RUNNER": str(REMOTE_RUNNER),
                    "MODEL": str(MODEL),
                    "REMOTE_MODEL": str(WORKER_MODEL),
                    "PACK": str(PACK),
                    "REMOTE_PACK": str(WORKER_PACK),
                    "SPEC": settings["spec"],
                    "TP_GRAPH": "1" if settings["graph"] == "on" else "0",
                    "SPEC_GRAPH": settings["spec_graph"],
                    "NO_PREFIX_CACHE": "1",
                    "HTTP_HOST": "127.0.0.1",
                    "HTTP_PORT": str(port),
                    "MASTER_PORT": str(master_port),
                    "REMOTE_PIDFILE": remote_pidfile,
                    "REMOTE_LOG": remote_log,
                    "REMOTE_STATUSFILE": remote_statusfile,
                    "PADDOCK_TP_SPAN_CAP": "512",
                    "PADDOCK_NO_PREFIX_CACHE": "1",
                })
                record = {
                    "label": label,
                    "config": config,
                    "workload": workload,
                    "repetition": rep,
                    "settings": settings,
                    "tp_span_cap": 512,
                    "unified": True,
                    "max_ctx": 65536,
                    "max_batch": 2,
                    "kv_dtype": "f16",
                    "temperature": 0,
                    "seed": 1,
                    "max_tokens": 512,
                    "prefix_cache_disabled_for_repeatability": True,
                    "master_port": master_port,
                    "http_port": port,
                }
                (raw_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n")
                print(f"START {label} graph={settings['graph']} spec={settings['spec']} port={port}", flush=True)
                with server_log.open("wb") as log:
                    proc = subprocess.Popen(["bash", str(LAUNCHER)], cwd=REPO,
                                            env=env, stdout=log, stderr=subprocess.STDOUT)
                    error = None
                    try:
                        wait_ready(port, proc, server_log)
                        client_log = raw_dir / "client.stdout"
                        cmd = [sys.executable, str(CLIENT), "--url", f"http://127.0.0.1:{port}",
                               "--prompt", str(PROMPTS[prompt_name]), "--outdir", str(raw_dir),
                               "--label", label, "--runs", "1", "--concurrency", str(concurrency)]
                        with client_log.open("wb") as output:
                            result = subprocess.run(cmd, cwd=REPO, env=env, stdout=output,
                                                    stderr=subprocess.STDOUT, check=False)
                        if result.returncode:
                            error = f"benchmark client exited {result.returncode}"
                    except Exception as exc:
                        error = repr(exc)
                    finally:
                        pid = runner_pid(master_port)
                        if pid:
                            (raw_dir / "coordinator.pid").write_text(str(pid) + "\n")
                            os.kill(pid, signal.SIGTERM)
                        else:
                            error = (error or "") + " coordinator PID not found for exact master port"
                        try:
                            launcher_rc = proc.wait(timeout=90)
                        except subprocess.TimeoutExpired:
                            remaining = runner_pid(master_port)
                            if remaining:
                                os.kill(remaining, signal.SIGKILL)
                            proc.terminate()
                            try:
                                launcher_rc = proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                launcher_rc = proc.wait(timeout=10)
                                error = (error or "") + " launcher required SIGKILL"
                        if launcher_rc:
                            error = (error or "") + f" launcher exit={launcher_rc}"
                subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "192.168.100.11:" + remote_log,
                                str(raw_dir / "worker.log")], check=False)
                status = subprocess.run(["ssh", "-o", "BatchMode=yes", "192.168.100.11",
                                         "cat", remote_statusfile], capture_output=True, text=True)
                (raw_dir / "worker.exit").write_text(status.stdout if status.returncode == 0 else
                                                    f"unavailable exit={status.returncode}: {status.stderr}")
                if error:
                    (raw_dir / "error.txt").write_text(error + "\n")
                    print(f"FAIL {label}: {error}", flush=True)
                    raise SystemExit(1)
                print(f"DONE {label}", flush=True)


if __name__ == "__main__":
    main()
