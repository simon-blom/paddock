#!/usr/bin/env python3
"""Isolated, reproducible Metal telemetry A/B and lifecycle qualification.

Never changes the user's library, endpoint configs or running services. Uses
loopback ports, a fresh temp data/runtime directory per block, no downloads,
and terminates only children it starts. Raw logs may contain ephemeral keys;
only measurement JSON is suitable for publishing. Run on each physical Mac.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import tempfile
import time
import urllib.request
import urllib.error

TEST_KEY = 'paddock-telemetry-local-test-only'


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def request(base, path, body=None, method=None, timeout=180):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
        headers={'Content-Type': 'application/json', 'Origin': base, 'Authorization': f'Bearer {TEST_KEY}'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        payload = json.load(error)
        detail = payload.get('error', {})
        # Before readiness, there are no request/customer secrets in our fresh
        # isolated service. Avoid echoing runner log tails from spawn failures.
        message = str(detail.get('message', detail) if isinstance(detail, dict) else detail).split('\n')[0]
        raise RuntimeError(f'{path}: HTTP {error.code}: {message[:400]}') from None


def wait_for(check, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except (OSError, ValueError):
            pass
        time.sleep(.2)
    raise RuntimeError('Timed out waiting for isolated test service')


def stop(child):
    if child and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def generation(base, runner_port, model_id, lane, count, wave):
    prompt = (f'Benchmark {wave}, lane {lane}. Write a detailed numbered list of 100 '
              'practical ways to improve a public library. Explain each item in two sentences.')
    body = {'model': model_id, 'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0, 'seed': 42, 'max_tokens': count, 'ignore_eos': True,
        'chat_template_kwargs': {'enable_thinking': False}, 'stream': True,
        'stream_options': {'include_usage': True}}
    req = urllib.request.Request(f'http://127.0.0.1:{runner_port}/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
            'Authorization': f'Bearer {TEST_KEY}'})
    start = time.perf_counter()
    events, parts, usage = [], [], None
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                break
            item = json.loads(payload)
            if item.get('error'):
                raise RuntimeError('Generation returned an error; inspect isolated log')
            if item.get('usage'):
                usage = item['usage']
            delta = (item.get('choices') or [{}])[0].get('delta', {})
            text = delta.get('content', '') or delta.get('reasoning_content', '')
            if text:
                events.append(time.perf_counter())
                parts.append(text)
    end = time.perf_counter()
    if not usage or not events or usage['completion_tokens'] != count:
        raise RuntimeError(f'Incomplete benchmark generation: usage={usage}')
    gaps = sorted((b-a)*1000 for a, b in zip(events, events[1:]))
    return {'seconds': end-start, 'ttft_s': events[0]-start,
        'tokens': usage['completion_tokens'], 'prompt_tokens': usage['prompt_tokens'],
        'p99_gap_ms': gaps[min(len(gaps)-1, int(len(gaps)*.99))] if gaps else None,
        'sha256': hashlib.sha256(''.join(parts).encode()).hexdigest()}


def wave(base, runner_port, concurrency, tokens, index):
    model_id = request(f'http://127.0.0.1:{runner_port}', '/v1/models')['data'][0]['id']
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        start = time.perf_counter()
        rows = list(pool.map(lambda lane: generation(base, runner_port, model_id, lane, tokens, index),
                             range(concurrency)))
        elapsed = time.perf_counter() - start
    return {'concurrency': concurrency, 'wall_s': elapsed,
        'output_tok_s': sum(r['tokens'] for r in rows)/elapsed, 'requests': rows}


def block(args, enabled, index):
    # macOS AF_UNIX paths must fit sun_path (104 bytes); the per-user TMPDIR
    # prefix alone is too long for the runner admin socket suffix.
    directory = Path(tempfile.mkdtemp(prefix='paddock-telemetry-ab-', dir='/tmp'))
    manager_port, runner_port = port(), port()
    base = f'http://127.0.0.1:{manager_port}'
    env = os.environ.copy()
    # Do not inherit serving overrides from the invoking shell.
    env = {k: v for k, v in env.items() if not k.startswith('PADDOCK_')}
    env.update(PADDOCK_DATA=str(directory), XDG_RUNTIME_DIR=str(directory/'runtime'),
        PADDOCK_RUNNER_BIN=str(args.runner), PADDOCK_DEVICE='metal',
        PADDOCK_METAL_TELEMETRY=str(int(enabled)), PADDOCK_TELEMETRY=str(int(enabled)))
    logs = open(directory/'manager.log', 'w')
    manager = subprocess.Popen([str(args.manager), '--host', '127.0.0.1', '--port', str(manager_port),
        '--model-dir', str(args.model.parent)], env=env, stdout=logs, stderr=subprocess.STDOUT)
    stream = None
    live_ports = set()
    result = {'enabled': enabled, 'block': index, 'waves': [], 'lifecycle': {},
              'local_diagnostics': str(directory)}
    try:
        wait_for(lambda: request(base, '/api/gpu', timeout=2))
        def spawn(p):
            value = request(base, '/api/runners', {'model': str(args.model), 'port': p,
                'host': '127.0.0.1', 'max_ctx': args.context, 'max_batch': 4,
                'spec': 'off', 'kv_cache_dtype': args.kv_cache_dtype,
                'vram_budget': args.budget_mib, 'persist': False, 'api_key': TEST_KEY})
            live_ports.add(p)
            return value['pid']
        pid = spawn(runner_port)
        if enabled:
            # Exercise real WS cadence and close handshake without an extra npm dependency.
            stream_log = open(directory/'stream.jsonl', 'w')
            stream = subprocess.Popen(['node', '-e', '''
const ws = new WebSocket(process.argv[1]);
ws.onmessage = e => process.stdout.write(e.data + '\\n');
ws.onerror = () => process.exit(3);
ws.onclose = e => process.exit(e.wasClean ? 0 : 1);
process.on('SIGTERM', () => { ws.close(); setTimeout(() => process.exit(2), 3000); });
''', base.replace('http:', 'ws:') + '/api/gpu/stream'], stdout=stream_log)
            wait_for(lambda: (request(base, '/api/gpu').get('reconciliation') or {}).get('runners'))
        for concurrency in (1, 4):
            wave(base, runner_port, concurrency, 32, -1)  # unmeasured warm-up
            for iteration in range(args.repeats):
                row = wave(base, runner_port, concurrency, args.tokens, iteration)
                row['gpu_after'] = request(base, '/api/gpu')
                result['waves'].append(row)
                (directory/'result.json').write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps({'block': index, 'enabled': enabled, 'c': concurrency,
                    'iteration': iteration, 'tok_s': round(row['output_tok_s'], 3)}), flush=True)
        if enabled and args.lifecycle:
            def observed(p, expected):
                value = request(base, '/api/gpu')
                recon = value.get('reconciliation') or {}
                runners = recon.get('runners', [])
                fresh = abs(time.time() - value.get('ts', 0)) < 5 and abs(time.time() - recon.get('ts', 0)) < 5
                return fresh and any(r['port'] == p and r['pid'] == expected and r.get('metal') for r in runners)
            # Co-resident runners, then repeated replacement on the same port.
            other = port()
            other_pid = spawn(other)
            wait_for(lambda: observed(runner_port, pid) and observed(other, other_pid))
            result['lifecycle']['two_runners'] = True
            request(base, f'/api/runners/{other}', method='DELETE')
            live_ports.remove(other)
            for _ in range(3):
                old = pid
                request(base, f'/api/runners/{runner_port}', method='DELETE')
                live_ports.remove(runner_port)
                pid = request(base, f'/api/servers/{runner_port}/start', {})['pid']
                live_ports.add(runner_port)
                assert pid != old
                wait_for(lambda: observed(runner_port, pid))
            result['lifecycle']['runner_restarts'] = 3
            # A suspended manager tests recovery from a sampling gap, NOT physical sleep.
            manager.send_signal(signal.SIGSTOP)
            time.sleep(13)
            manager.send_signal(signal.SIGCONT)
            wait_for(lambda: observed(runner_port, pid))
            result['lifecycle']['sampling_pause_resume_seconds'] = 13
            result['lifecycle']['physical_sleep_wake'] = 'not tested'
        return result
    finally:
        stop(stream)
        if stream:
            result['lifecycle']['websocket_close_clean'] = stream.returncode == 0
        (directory/'result.json').write_text(json.dumps(result, indent=2) + '\n')
        if manager.poll() is None:
            manager.send_signal(signal.SIGCONT)
            for p in live_ports:
                try:
                    request(base, f'/api/runners/{p}', method='DELETE')
                except (OSError, RuntimeError):
                    pass
        stop(manager)
        logs.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manager', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--budget-mib', type=int, default=32768)
    parser.add_argument('--kv-cache-dtype', default='auto')
    parser.add_argument('--order', default='off,on,on,off')
    parser.add_argument('--lifecycle', action='store_true')
    parser.add_argument('--resident-pairs', action='store_true',
        help='Matched resident runners, alternating paired waves to control thermal drift; manager sampler remains active for both')
    args = parser.parse_args()
    if any(value not in ('on', 'off') for value in args.order.split(',')):
        parser.error('--order must be comma-separated on/off values')
    if args.resident_pairs:
        resident_pairs(args)
        return
    result = {'order': [value == 'on' for value in args.order.split(',')], 'tokens': args.tokens,
        'model': args.model.name, 'context': args.context, 'budget_mib': args.budget_mib,
        'machine': subprocess.check_output(['sysctl', '-n', 'hw.model'], text=True).strip(),
        'os': subprocess.check_output(['sw_vers', '-productVersion'], text=True).strip(),
        'runner_sha256': hashlib.sha256(args.runner.read_bytes()).hexdigest(), 'blocks': []}
    for index, enabled in enumerate(result['order']):
        result['blocks'].append(block(args, enabled, index))
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    for concurrency in (1, 4):
        medians = {}
        for enabled in (False, True):
            rows = [w['output_tok_s'] for b in result['blocks'] if b['enabled'] == enabled
                    for w in b['waves'] if w['concurrency'] == concurrency]
            medians[enabled] = statistics.median(rows) if rows else None
        if all(medians.values()):
            print(json.dumps({'c': concurrency, 'off': medians[False], 'on': medians[True],
                              'delta_pct': (medians[True]/medians[False]-1)*100}), flush=True)


def resident_pairs(args):
    """Separate runner-fence overhead from thermal drift across process restarts.

    Both copies are resident with matched budgets/contexts; only one infers at a time.
    Manager sampling stays active for both, so this experiment measures the
    runner's added timing/allocation recording, not the full manager pipeline.
    """
    services = []
    result = {'mode': 'resident-pairs', 'waves': [], 'model': args.model.name,
        'context': args.context, 'budget_mib': args.budget_mib, 'tokens': args.tokens}
    try:
        for enabled in (False, True):
            directory = Path(tempfile.mkdtemp(prefix='paddock-telemetry-pair-', dir='/tmp'))
            mp, rp = port(), port()
            base = f'http://127.0.0.1:{mp}'
            env = {k: v for k, v in os.environ.items() if not k.startswith('PADDOCK_')}
            env.update(PADDOCK_DATA=str(directory), XDG_RUNTIME_DIR=str(directory/'runtime'),
                PADDOCK_RUNNER_BIN=str(args.runner), PADDOCK_DEVICE='metal',
                PADDOCK_METAL_TELEMETRY=str(int(enabled)), PADDOCK_TELEMETRY='1')
            log = open(directory/'manager.log', 'w')
            child = subprocess.Popen([str(args.manager), '--host', '127.0.0.1', '--port', str(mp),
                '--model-dir', str(args.model.parent)], env=env, stdout=log, stderr=subprocess.STDOUT)
            services.append((base, rp, child, log))
            wait_for(lambda: request(base, '/api/gpu', timeout=2))
            request(base, '/api/runners', {'model': str(args.model), 'port': rp, 'host': '127.0.0.1',
                'max_ctx': args.context, 'max_batch': 4, 'spec': 'off', 'kv_cache_dtype': args.kv_cache_dtype,
                'vram_budget': args.budget_mib, 'persist': False, 'api_key': TEST_KEY})
        for base, rp, _, _ in services:
            wave(base, rp, 1, 128, -2)
            wave(base, rp, 4, 128, -2)
        for concurrency in (1, 4):
            for iteration in range(args.repeats):
                for enabled in ((False, True) if iteration % 2 == 0 else (True, False)):
                    base, rp, _, _ = services[int(enabled)]
                    row = wave(base, rp, concurrency, args.tokens, iteration)
                    row.update(enabled=enabled, iteration=iteration, gpu_after=request(base, '/api/gpu'))
                    result['waves'].append(row)
                    args.output.write_text(json.dumps(result, indent=2) + '\n')
                    print(json.dumps({'paired': True, 'enabled': enabled, 'c': concurrency,
                        'iteration': iteration, 'tok_s': round(row['output_tok_s'], 3)}), flush=True)
    finally:
        for base, rp, child, log in services:
            try:
                request(base, f'/api/runners/{rp}', method='DELETE')
            except (OSError, RuntimeError):
                pass
            stop(child)
            log.close()


if __name__ == '__main__':
    main()
