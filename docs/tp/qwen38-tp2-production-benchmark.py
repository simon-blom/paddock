#!/usr/bin/env python3
"""Reproducible streaming chat benchmark for Qwen3.8 TP2 serving."""
import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def scrape_spec_metrics(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=5) as response:
            text = response.read().decode("utf-8", errors="replace")
        values = {"drafted": 0, "accepted": 0}
        for line in text.splitlines():
            if line.startswith("paddock_spec_decode_draft_tokens_total "):
                values["drafted"] = int(float(line.rsplit(" ", 1)[1]))
            elif line.startswith("paddock_spec_decode_accepted_tokens_total "):
                values["accepted"] = int(float(line.rsplit(" ", 1)[1]))
        return values
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def stream_request(url, model, prompt, run_id, start_gate=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0,
        "seed": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if start_gate:
        start_gate.wait()
    start = time.monotonic()
    first_token = None
    chunks = []
    usage = None
    finish_reason = None
    raw = bytearray()
    status = None
    try:
        with urllib.request.urlopen(req, timeout=1200) as response:
            status = response.status
            for line in response:
                raw.extend(line)
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    text = choice.get("delta", {}).get("content")
                    if text:
                        if first_token is None:
                            first_token = time.monotonic()
                        chunks.append(text)
        end = time.monotonic()
        content = "".join(chunks)
        completion_tokens = (usage or {}).get("completion_tokens")
        prompt_tokens = (usage or {}).get("prompt_tokens")
        decode_wall = end - first_token if first_token else None
        return {
            "run_id": run_id,
            "http_status": status,
            "client_wall_s": end - start,
            "ttft_s": first_token - start if first_token else None,
            "stream_decode_wall_s": decode_wall,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "completion_reason": finish_reason,
            "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "output": content,
            "raw_sse": raw.decode("utf-8", errors="replace"),
            "client_observed_prefill_tok_s": (
                prompt_tokens / (first_token - start)
                if prompt_tokens and first_token and first_token > start
                else None
            ),
            "client_observed_output_tok_s": (
                completion_tokens / decode_wall
                if completion_tokens and decode_wall and decode_wall > 0
                else None
            ),
            "client_started_monotonic": start,
            "client_finished_monotonic": end,
        }
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {
            "run_id": run_id,
            "http_status": status,
            "error": repr(exc),
            "client_started_monotonic": start,
            "client_finished_monotonic": time.monotonic(),
            "raw_sse": raw.decode("utf-8", errors="replace"),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:11540")
    parser.add_argument("--model", default="Qwen3.8-27B-UD-Q4_K_M")
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.runs < 1 or args.concurrency not in (1, 2):
        parser.error("runs must be positive and concurrency must be 1 or 2")
    prompt = args.prompt.read_text(encoding="utf-8")
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / f"{args.label}.jsonl"
    for rep in range(args.runs):
        metrics_before = scrape_spec_metrics(args.url)
        gate = threading.Barrier(args.concurrency) if args.concurrency > 1 else None
        results = [None] * args.concurrency
        def task(i):
            results[i] = stream_request(args.url, args.model, prompt,
                                        f"{args.label}-r{rep + 1}-q{i + 1}", gate)
        threads = [threading.Thread(target=task, args=(i,), daemon=True)
                   for i in range(args.concurrency)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        metrics_after = scrape_spec_metrics(args.url)
        spec_delta = None
        if metrics_before is not None and metrics_after is not None:
            spec_delta = {
                "drafted": metrics_after["drafted"] - metrics_before["drafted"],
                "accepted": metrics_after["accepted"] - metrics_before["accepted"],
            }
        valid = [r for r in results if r]
        for result in valid:
            result["spec_counter_delta"] = spec_delta
            drafted = spec_delta["drafted"] if spec_delta else 0
            accepted = spec_delta["accepted"] if spec_delta else 0
            result["spec_draft_acceptance_rate"] = accepted / drafted if drafted else None
            result["accepted_token_tok_s_client_decode"] = (
                accepted / result["stream_decode_wall_s"]
                if accepted and result.get("stream_decode_wall_s") else None
            )
            completion = result.get("completion_tokens") or 0
            estimated_steps = max(completion - accepted, 0)
            result["estimated_target_verify_steps"] = estimated_steps if drafted else None
            result["mean_accepted_drafts_per_step_estimate"] = (
                accepted / estimated_steps if drafted and estimated_steps else None
            )
        for result in valid:
            stem = result["run_id"]
            (args.outdir / f"{stem}.sse").write_text(result.pop("raw_sse"), encoding="utf-8")
            (args.outdir / f"{stem}.txt").write_text(result.pop("output", ""), encoding="utf-8")
            with summary_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
        if len(valid) != args.concurrency or any(r.get("http_status") != 200 for r in valid):
            raise SystemExit(f"request failure: {valid}")
        if args.concurrency == 2:
            starts = [r["client_started_monotonic"] for r in valid]
            ends = [r["client_finished_monotonic"] for r in valid]
            overlap_s = min(ends) - max(starts)
            if overlap_s <= 0:
                raise SystemExit(f"requests did not overlap: {valid}")
            tokens = [r.get("completion_tokens") or 0 for r in valid]
            aggregate = {
                "label": args.label,
                "rep": rep + 1,
                "concurrency": 2,
                "overlap_s": overlap_s,
                "aggregate_wall_s": max(ends) - min(starts),
                "aggregate_output_tokens": sum(tokens),
                "aggregate_output_tok_s": sum(tokens) / (max(ends)-min(starts)),
                "completion_tokens": tokens,
                "spec_counter_delta": spec_delta,
                "spec_draft_acceptance_rate": (
                    spec_delta["accepted"] / spec_delta["drafted"]
                    if spec_delta and spec_delta["drafted"] else None
                ),
                "accepted_token_tok_s_aggregate": (
                    spec_delta["accepted"] / (max(ends)-min(starts))
                    if spec_delta and max(ends) > min(starts) else None
                ),
            }
            with summary_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"aggregate": aggregate}, sort_keys=True) + "\n")
            print(json.dumps(aggregate, sort_keys=True), flush=True)
        else:
            print(json.dumps({k: valid[0].get(k) for k in (
                "run_id", "http_status", "client_wall_s", "ttft_s", "prompt_tokens",
                "completion_tokens", "client_observed_prefill_tok_s",
                "client_observed_output_tok_s", "completion_reason", "output_sha256")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
