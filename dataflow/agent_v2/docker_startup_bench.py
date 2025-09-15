#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Docker container startup benchmark with resource configs.
- Measures cold vs warm start latency with millisecond precision.
- Handles full lifecycle: create -> start -> poll running -> stop -> remove.
- Can include image pull time for "true cold start".
- Supports multiple resource configurations and repeated trials.
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
from typing import Dict, Any, List, Optional

import docker
from docker.errors import NotFound, APIError, ImageNotFound

# ---------- Timing helpers ----------

def now_ns() -> int:
    return time.perf_counter_ns()

def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0

# ---------- Docker helpers ----------

def ensure_client() -> docker.DockerClient:
    # Uses environment (DOCKER_HOST / local socket)
    return docker.from_env()

def image_present(client: docker.DockerClient, image: str) -> bool:
    try:
        client.images.get(image)
        return True
    except ImageNotFound:
        return False

def remove_image_if_exists(client: docker.DockerClient, image: str) -> None:
    try:
        client.images.remove(image=image, force=True, noprune=False)
    except ImageNotFound:
        pass
    except APIError as e:
        print(f"[WARN] Failed to remove image {image}: {e}", file=sys.stderr)

def pull_image_timed(client: docker.DockerClient, image: str) -> int:
    t0 = now_ns()
    # docker-py pull returns an Image object; stream not needed for timing granularity
    client.images.pull(image)
    t1 = now_ns()
    return t1 - t0

def wait_running(container, timeout_s: float = 30.0, poll_interval_s: float = 0.05) -> bool:
    """
    Poll container.status until 'running' or timeout.
    Returns True if reached running, else False.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        container.reload()
        if getattr(container, "status", None) == "running":
            return True
        time.sleep(poll_interval_s)
    return False

def safe_stop_and_remove(container, timeout: int = 2):
    try:
        container.reload()
    except Exception:
        pass
    try:
        container.stop(timeout=timeout)
    except Exception:
        pass
    try:
        container.remove(force=True)
    except Exception:
        pass

# ---------- Resource config helpers ----------

def preset_resource_configs() -> Dict[str, Dict[str, Any]]:
    """
    Returns a few example resource configs.
    Note:
      - nano_cpus: 1e9 = 1 CPU (Docker uses 1e9 as 100% of a single CPU)
      - mem_limit: strings like "256m", "1g"
      - cpuset_cpus: pin to given CPU cores, e.g. "0", "0-1"
    """
    return {
        "minimal":   {"nano_cpus": int(0.5e9), "mem_limit": "128m", "cpu_shares": 128},
        "balanced":  {"nano_cpus": int(1.0e9), "mem_limit": "512m", "cpu_shares": 512},
        "highcpu":   {"nano_cpus": int(2.0e9), "mem_limit": "512m", "cpu_shares": 1024},
        "highmem":   {"nano_cpus": int(1.0e9), "mem_limit": "2g",   "cpu_shares": 512},
        "pinned":    {"nano_cpus": int(1.0e9), "mem_limit": "512m", "cpu_shares": 512, "cpuset_cpus": "0"},
    }

def merge_resource_args(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = base.copy()
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged

# ---------- Benchmark core ----------

def run_single_trial(
    client: docker.DockerClient,
    image: str,
    command: Optional[str],
    resources: Dict[str, Any],
    include_pull: bool,
    cold_remove_image: bool,
    start_timeout_s: float,
) -> Dict[str, Any]:
    """
    Returns timing dict:
      pull_ms, create_ms, start_ms, to_running_ms, total_ms
    """
    timings = {
        "pull_ms": 0.0,
        "create_ms": 0.0,
        "start_ms": 0.0,
        "to_running_ms": 0.0,
        "total_ms": 0.0,
    }

    # ---- Cold-start preparation (optional) ----
    if cold_remove_image:
        remove_image_if_exists(client, image)

    # ---- Pull (optional timing) ----
    if include_pull:
        if not image_present(client, image):
            pull_ns = pull_image_timed(client, image)
            timings["pull_ms"] = ns_to_ms(pull_ns)
        else:
            # Image already present; if include_pull==True but present, record 0 ms
            timings["pull_ms"] = 0.0
    else:
        # Ensure image exists for create/start to succeed
        if not image_present(client, image):
            client.images.pull(image)

    # ---- Create ----
    t0 = now_ns()
    container = client.containers.create(
        image=image,
        command=command,
        detach=True,
        tty=False,
        stdin_open=False,
        # Resource limits:
        nano_cpus=resources.get("nano_cpus"),
        mem_limit=resources.get("mem_limit"),
        cpu_shares=resources.get("cpu_shares"),
        cpuset_cpus=resources.get("cpuset_cpus"),
        # You may also set blkio_weight, device_cgroup_rules, etc., if needed
        name=None  # auto-generated
    )
    t1 = now_ns()
    timings["create_ms"] = ns_to_ms(t1 - t0)

    # ---- Start ----
    started_ok = False
    t2 = now_ns()
    try:
        container.start()
        started_ok = True
    finally:
        t3 = now_ns()
        timings["start_ms"] = ns_to_ms(t3 - t2)

    # ---- Wait until running (status) ----
    to_running_ms = 0.0
    if started_ok:
        t4 = now_ns()
        reached = wait_running(container, timeout_s=start_timeout_s)
        t5 = now_ns()
        to_running_ms = ns_to_ms(t5 - t4)
        timings["to_running_ms"] = to_running_ms
    else:
        timings["to_running_ms"] = float("nan")

    # ---- Total ----
    timings["total_ms"] = (
        timings["pull_ms"] + timings["create_ms"] + timings["start_ms"] + timings["to_running_ms"]
    )

    # ---- Cleanup lifecycle ----
    safe_stop_and_remove(container, timeout=2)

    return timings

def bench_mode(
    client: docker.DockerClient,
    image: str,
    command: Optional[str],
    mode: str,  # "cold" or "warm"
    resources: Dict[str, Any],
    trials: int,
    include_pull: bool,
    start_timeout_s: float,
) -> List[Dict[str, Any]]:
    results = []
    for i in range(trials):
        cold_remove = (mode == "cold")
        rec = run_single_trial(
            client=client,
            image=image,
            command=command,
            resources=resources,
            include_pull=include_pull,
            cold_remove_image=cold_remove,
            start_timeout_s=start_timeout_s,
        )
        rec.update({
            "mode": mode,
            "trial": i + 1,
            "resource_name": resources.get("_name", "custom"),
            "nano_cpus": resources.get("nano_cpus"),
            "mem_limit": resources.get("mem_limit"),
            "cpu_shares": resources.get("cpu_shares"),
            "cpuset_cpus": resources.get("cpuset_cpus"),
            "image": image,
            "command": command or "",
        })
        results.append(rec)
    return results

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    import statistics as stats
    if not results:
        return {}
    keys = ["pull_ms", "create_ms", "start_ms", "to_running_ms", "total_ms"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in results if isinstance(r[k], (int, float)) and not (r[k] != r[k])]  # filter NaN
        if not vals:
            continue
        summary[k] = {
            "mean": stats.mean(vals),
            "median": stats.median(vals),
            "p95": stats.quantiles(vals, n=20)[18],  # ~95th
            "min": min(vals),
            "max": max(vals),
            "count": len(vals),
        }
    return summary

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(sorted({k for r in rows for k in r.keys()}))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[INFO] Saved CSV to: {os.path.abspath(path)}")

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Docker container startup latency.")
    p.add_argument("--image", type=str, default="alpine:3.20", help="Docker image (e.g., alpine:3.20)")
    p.add_argument("--command", type=str, default="sleep 5", help="Container command (quoted string)")

    p.add_argument("--cold", type=str, default="true", help="Run cold start tests (true/false)")
    p.add_argument("--warm", type=str, default="true", help="Run warm start tests (true/false)")
    p.add_argument("--include-pull", type=str, default="true",
                   help="For cold: include image pull time in total (true/false)")

    p.add_argument("--trials", type=int, default=5, help="Trials per (mode,resource)")

    p.add_argument("--resource-presets", type=str, default="minimal,balanced,highcpu,highmem,pinned",
                   help="Comma-separated presets to test (from: minimal,balanced,highcpu,highmem,pinned)")
    p.add_argument("--resource-json", type=str, default="",
                   help='Optional custom resource config JSON list, '
                        'e.g. \'[{"_name":"r1","nano_cpus":1000000000,"mem_limit":"512m"}]\'')

    p.add_argument("--start-timeout-s", type=float, default=30.0, help="Timeout waiting for running status")
    p.add_argument("--save-csv", type=str, default="", help="Path to save detailed per-trial CSV results")
    return p.parse_args()

def str2bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "y", "t")

def main():
    args = parse_args()

    # Graceful Ctrl-C
    def handle_sigint(signum, frame):
        print("\n[INFO] Interrupted, exiting...", file=sys.stderr)
        sys.exit(1)
    signal.signal(signal.SIGINT, handle_sigint)

    client = ensure_client()

    # Assemble resource sets
    presets = preset_resource_configs()
    selected_names = [x.strip() for x in args.resource_presets.split(",") if x.strip()]
    resource_sets: List[Dict[str, Any]] = []
    for name in selected_names:
        if name in presets:
            cfg = presets[name].copy()
            cfg["_name"] = name
            resource_sets.append(cfg)
        else:
            print(f"[WARN] Unknown preset '{name}', skipping.", file=sys.stderr)

    if args.resource_json:
        try:
            custom = json.loads(args.resource_json)
            if isinstance(custom, dict):
                custom = [custom]
            for cfg in custom:
                if "_name" not in cfg:
                    cfg["_name"] = "custom"
                resource_sets.append(cfg)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse --resource-json: {e}", file=sys.stderr)
            sys.exit(2)

    if not resource_sets:
        print("[ERROR] No resource configs selected.", file=sys.stderr)
        sys.exit(2)

    image = args.image
    command = args.command if args.command else None

    will_run_cold = str2bool(args.cold)
    will_run_warm = str2bool(args.warm)
    include_pull = str2bool(args.include_pull)

    all_rows: List[Dict[str, Any]] = []

    # Warm pre-pull to stabilize warm runs (and to allow cold without include_pull=False case)
    if will_run_warm or not include_pull:
        try:
            if not image_present(client, image):
                print(f"[INFO] Pulling image for warm/prep: {image}")
                client.images.pull(image)
        except APIError as e:
            print(f"[ERROR] Failed to pull image {image}: {e}", file=sys.stderr)
            sys.exit(3)

    # Execute benchmarks
    for res in resource_sets:
        if will_run_cold:
            print(f"[INFO] Running COLD trials for {res.get('_name')} ...")
            cold_rows = bench_mode(
                client=client,
                image=image,
                command=command,
                mode="cold",
                resources=res,
                trials=args.trials,
                include_pull=include_pull,
                start_timeout_s=args.start_timeout_s,
            )
            all_rows.extend(cold_rows)

        if will_run_warm:
            print(f"[INFO] Running WARM trials for {res.get('_name')} ...")
            warm_rows = bench_mode(
                client=client,
                image=image,
                command=command,
                mode="warm",
                resources=res,
                trials=args.trials,
                include_pull=False,  # warm start never includes pull
                start_timeout_s=args.start_timeout_s,
            )
            all_rows.extend(warm_rows)

    # Summaries per (mode,resource)
    from collections import defaultdict
    group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        key = f"{r['mode']}::{r['resource_name']}"
        group[key].append(r)

    print("\n========== SUMMARY (ms) ==========")
    for key, rows in group.items():
        summ = summarize(rows)
        print(f"\n[{key}] N={len(rows)}")
        for metric, stat in summ.items():
            print(f"  {metric:13s}: mean={stat['mean']:.2f}  p95={stat['p95']:.2f}  "
                  f"median={stat['median']:.2f}  min={stat['min']:.2f}  max={stat['max']:.2f}")

    if args.save_csv:
        write_csv(args.save_csv, all_rows)

if __name__ == "__main__":
    main()


# python docker_startup_bench.py \
#   --image alpine:3.20 \
#   --command "sleep 5" \
#   --trials 5 \
#   --cold true \
#   --include-pull true \
#   --warm true \
#   --save-csv startup_results.csv
