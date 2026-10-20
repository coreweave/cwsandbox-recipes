"""The dashboard's log parser: launcher, Gym and NeMo-RL lines become page events."""

import json
import socket
import urllib.request

from async_grpo_tool_use.dashboard import server
from async_grpo_tool_use.dashboard.server import Hub, LineParser

STEP_LOG = """\
[grpo] run tag async-grpo-1a2b3c4d
[grpo] inference: gateway gw-1 ready at https://async-grpo.gw.example.com
[grpo] sunk: node h100-001, run dir /mnt/data/me/async-grpo-tool-use/runs/x
[gym] (math_sandbox_eval) [math_tir_sandbox] starting 256 episode sandboxes
[grpo] calibration: boxed_correct   100% graded as expected (79 cases)
========================= Step 1/2 =========================
▶ Preparing batch...
▶ Generating responses for batch of size 128...
[grpo] hot-load v1: exporting weights
[grpo] hot-load v1: uploading and loading
[grpo] hot-load v1: 339 tensors, 14.2 GiB; export 33s, upload and load 64s
[gym] (policy_model) @@metric {"ts": 1.0, "k": "sampler", "inflight": 12, "done": 3, "prompt_tokens": 90, "gen_tokens": 4000}
▶ Processing rewards...,
▶ Computing logprobs...
▶ Training policy...

📊 Training Results:
  • Loss: 0.0123
  • Generation KL Error: 0.0005
  • Avg Filtered Reward: -0.2500
  • Avg Total Reward: -0.5000
  • Mean Generation Length: 812.5000

⏱️  Timing:
  • Total step time: 129.75s
  • generation: 60.10s (46.3%)
  • Throughputs (per GPU):
    - E2E (Tokens/sec/gpu): 300.00
    - Generation Worker Group (Tokens/sec/gpu): 4000.00
  • Throughputs (per Group):
    - E2E (Tokens/sec): 2400.00
    - Training Worker Group (Tokens/sec): 21000.00
    - Generation Worker Group (Tokens/sec): 8000.00
"""


def parse(text):
    hub = Hub()
    parser = LineParser(hub)
    for line in text.splitlines(keepends=True):
        parser.feed(line)
    return hub.events


def test_step_log_becomes_stages_phases_and_step_metrics():
    events = parse(STEP_LOG)
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == ["run", "gateway", "node", "pool", "calibration"]
    phases = [e["phase"] for e in events if e["type"] == "phase"]
    assert phases == ["generation", "export", "hotload", "generation", "rewards", "logprobs", "training"]
    [hotload] = [e for e in events if e["type"] == "hotload"]
    assert (hotload["version"], hotload["gib"], hotload["load_s"]) == (1, 14.2, 64)
    [done] = [e for e in events if e["type"] == "step_done"]
    assert done["step"] == 1
    assert done["metrics"] == {
        "loss": 0.0123, "kl_error": 0.0005, "reward": -0.25, "total_reward": -0.5, "gen_length": 812.5,
        "step_time": 129.75, "e2e_tokens_s": 2400.0, "train_tokens_s": 21000.0, "gen_tokens_s": 8000.0,
    }  # fmt: skip
    assert done["timings"]["generation"] == 60.10


def test_metric_lines_pass_through_and_skip_the_log():
    events = parse(STEP_LOG)
    [sampler] = [e for e in events if e["type"] == "metric"]
    assert (sampler["k"], sampler["inflight"], sampler["gen_tokens"]) == ("sampler", 12, 4000)
    assert not any("@@metric" in e.get("line", "") for e in events)


def test_example_rows_from_the_episode_server_reach_the_page_whole():
    row = {"ts": 2.0, "k": "example", "prompt": "What is 6*7?", "cells": 1, "answer": "42", "expected": "42",
           "reward": 1.0, "sandbox": "sbx-1",
           "turns": [{"text": "Compute it.", "code": "print(6*7)\n", "cell": "/work/cell_1.py", "output": "42\n"}]}  # fmt: skip
    [event] = parse(f"[gym] (math_sandbox_eval) @@metric {json.dumps(row)}\n")
    assert event == {"t": event["t"], "type": "metric", **row}


def test_a_background_hot_load_that_ends_during_training_keeps_the_training_phase():
    events = parse("""\
========================= Step 2/3 =========================
[grpo] hot-load v2: exporting weights
[grpo] hot-load v2: uploading and loading
[gym] (policy_model) @@metric {"k": "hotload", "version": 2, "state": "start", "ts": 5.0}
========================= Step 3/3 =========================
▶ Training policy...
[grpo] hot-load v2: 339 tensors, 14.2 GiB; export 31s, upload and load 60s
@@metric {"k": "hotload", "version": 2, "state": "done", "ts": 66.0}
""")
    assert [e["phase"] for e in events if e["type"] == "phase"] == ["export", "hotload", "training"]
    states = [(e["version"], e["state"]) for e in events if e.get("k") == "hotload"]
    assert states == [(2, "start"), (2, "done")]


ASYNC_STEP_LOG = """\
========================= Step 2/3 =========================
📦 Sampling from replay buffer...
[grpo] hot-load v2: 339 tensors, 14.2 GiB; export 4s, upload and load 70s (upload 30s, swap 40s; trainer 1 ahead), in the background
[grpo] hot-load v3: skipped, v4 is newer
@@metric {"k": "hotload", "version": 2, "state": "done", "ts": 1700000100.0}
▶ Training policy...
[grpo] hot-load v3: exporting weights
[grpo] hot-load v3: uploading and loading while training continues
@@metric {"k": "hotload", "version": 3, "state": "start", "ts": 1700000150.0}
"""


def test_background_hot_load_is_on_the_timeline_but_is_not_a_trainer_phase():
    events = parse(ASYNC_STEP_LOG)
    phases = [e["phase"] for e in events if e["type"] == "phase"]
    # The trainer trains while the hot-load runs, so neither hot-load line changes its phase.
    assert phases == ["generation", "training", "export"]
    [done] = [e for e in events if e["type"] == "hotload"]
    assert (done["version"], done["load_s"]) == (2, 70)
    timeline = [(e["version"], e["state"], e["ts"]) for e in events if e.get("k") == "hotload"]
    assert timeline == [(2, "done", 1700000100.0), (3, "start", 1700000150.0)]
    assert all(e["type"] == "metric" for e in events if e.get("k") == "hotload")


def test_the_local_page_moves_to_a_free_port_when_its_port_is_taken():
    """Another dashboard already on the port must not stop the run after the viewer exists."""
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    port = taken.getsockname()[1]
    try:
        url = server.serve(Hub(), {}, port)
        assert url.startswith("http://localhost:") and not url.endswith(f":{port}")
        with urllib.request.urlopen(url.replace("localhost", "127.0.0.1") + "/meta", timeout=5) as response:
            assert response.status == 200
    finally:
        taken.close()
