"""Application-managed Agents API recipe. Running it provisions billable resources."""

import argparse
import csv
import io
import json
import math
import os
import queue
import re
import shlex
import signal
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from cwsandbox import AuthStrategy, ResourceOptions, Sandbox
from cwsandbox.exceptions import SandboxNotFoundError
from dotenv import load_dotenv
from openai import NotFoundError, OpenAI

ROOT = Path(__file__).resolve().parent
WORKSPACE = "/workspace/project"
EXECUTOR_VERSION = "0.155.0-alpha.6"
INPUTS = ("TASK.md", "fixtures/requests.csv", "fixtures/capacity.csv", "fixtures/deployments.json")
ARTIFACTS = (
    "latency.json",
    "latency.py",
    "errors.json",
    "errors.py",
    "capacity.json",
    "capacity.py",
    "incident.md",
    "followup.md",
)
KEYS = ("CWSANDBOX_API_KEY", "WANDB_API_KEY", "OPENAI_API_KEY", "OPENAI_EXECUTOR_API_KEY")
AUTH_MODES = {"coreweave": AuthStrategy.COREWEAVE_API_KEY, "wandb": AuthStrategy.WANDB}
AUTH_KEYS = {"coreweave": "CWSANDBOX_API_KEY", "wandb": "WANDB_API_KEY"}
INSTRUCTIONS = """You coordinate an investigation of a synthetic inference-service incident.
Read TASK.md and follow its metric definitions and output contracts exactly. Use the native
subagent tools to create three distinct specialists named latency, errors, and capacity.
Give them independent tasks with literal SPECIALIST=latency, SPECIALIST=errors, and
SPECIALIST=capacity instruction markers, and create all three before waiting. Each specialist must
execute its own analysis script and write only its assigned output/{name}.py and .json.
All agents share /workspace/project. Wait for all three, inspect their evidence, and write
output/incident.md yourself. Do not replace delegation with local subprocesses or threads.
Never inspect environment variables or credentials. Use only the provided synthetic inputs.
Do not claim concurrency unless the runtime evidence establishes it.
"""
FOLLOWUP = """Continue this same investigation using the existing specialist findings and
output/incident.md. Write output/followup.md following the follow-up instructions in TASK.md.
Do not recreate the input data, delegate again, or replace specialist artifacts.
"""


def now():
    return datetime.now(UTC).isoformat()


def redact(text):
    for name in KEYS:
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return text


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(redact(json.dumps(value, indent=2)) + "\n")
    temporary.replace(path)


def remaining(deadline, cap=30):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("Recipe runtime deadline reached")
    return min(cap, left)


def inputs(root=ROOT):
    # Explicit paths prevent uploading credentials, the verifier, or unrelated project files.
    result = {}
    for name in INPUTS:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlinked recipe input: {name}")
        result[name] = path.read_bytes()
    validate_inputs(result)
    return result


def validate_inputs(payloads):
    if not payloads["TASK.md"].decode().strip():
        raise ValueError("TASK.md is empty")
    deployments = json.loads(payloads["fixtures/deployments.json"])
    if deployments.get("synthetic") is not True or not deployments.get("deployments"):
        raise ValueError("Expected synthetic deployment fixtures")
    datetime.fromisoformat(deployments["cutoff"])
    schemas = {
        "requests.csv": {
            "request_id",
            "timestamp",
            "region",
            "deployment_id",
            "status_code",
            "latency_ms",
        },
        "capacity.csv": {"timestamp", "region", "replicas", "utilization_pct", "queue_depth"},
    }
    for name, columns in schemas.items():
        reader = csv.DictReader(io.StringIO(payloads["fixtures/" + name].decode()))
        if set(reader.fieldnames or ()) != columns:
            raise ValueError(f"Unexpected columns in {name}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty fixture: {name}")
        for row in rows:
            if any(value is None or not value.strip() for value in row.values()):
                raise ValueError(f"Missing fixture value in {name}")
            datetime.fromisoformat(row["timestamp"])
            for field in columns & {
                "status_code",
                "latency_ms",
                "replicas",
                "utilization_pct",
                "queue_depth",
            }:
                number = float(row[field])
                if not math.isfinite(number) or number < 0:
                    raise ValueError(f"Invalid {field} in {name}")


def environment(session):
    env = session.environment
    if env is None or env.type != "self_hosted" or env.workspace_directory != WORKSPACE:
        raise ValueError("Unexpected Agents API environment")
    url = urlsplit(env.remote_url)
    if (
        url.scheme not in ("wss", "https")
        or url.hostname not in ("api.openai.com", "codex-cloud-environments.chatgpt.com")
        or url.username
        or url.password
        or url.port not in (None, 443)
    ):
        raise ValueError("Unexpected executor endpoint")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", env.id):
        raise ValueError("Invalid environment ID")
    return env


def execute(sandbox, command, deadline, cap=60, *, stage="Sandbox setup"):
    timeout = math.ceil(remaining(deadline, cap))
    result = sandbox.exec(command, timeout_seconds=timeout).result(timeout=timeout + 5)
    if result.returncode != 0:
        # Redact before truncating so the tail cannot expose part of a credential.
        stdout = redact(result.stdout)[-2000:] or "(empty)"
        stderr = redact(result.stderr)[-2000:] or "(empty)"
        raise RuntimeError(
            f"{stage} failed (exit code {result.returncode}). "
            f"Last 2000 characters of each output stream:\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return result


def start_executor(sandbox, env, deadline):
    bootstrap = (
        "set -eu\n"
        f"mkdir -p {WORKSPACE}/fixtures {WORKSPACE}/output /workspace/home\n"
        "apt-get update -qq\n"
        "apt-get install -y -qq --no-install-recommends python3\n"
        f"npm install --global @openai/codex@{EXECUTOR_VERSION}\n"
        "codex exec-server --help >/dev/null\n"
    )
    execute(sandbox, ["sh", "-c", bootstrap], deadline, cap=180, stage="Sandbox bootstrap")
    command = (
        f"cd {WORKSPACE}\n"
        f"nohup codex exec-server --remote {shlex.quote(env.remote_url)} "
        f"--environment-id {shlex.quote(env.id)} "
        ">/workspace/executor.log 2>&1 </dev/null &"
    )
    execute(sandbox, ["sh", "-c", command], deadline, stage="Executor launch")


def events_until(stream, deadline):
    # A wall deadline must also work while an SSE stream is silent or sending heartbeats.
    inbox = queue.Queue()
    stopped = threading.Event()

    def read():
        try:
            for event in stream:
                if stopped.is_set():
                    break
                inbox.put(event)
        except Exception as error:
            inbox.put(error)
        finally:
            inbox.put(None)

    threading.Thread(target=read, daemon=True).start()
    try:
        while True:
            try:
                event = inbox.get(timeout=remaining(deadline, 600))
            except queue.Empty:
                raise TimeoutError("No completed root turn before runtime deadline") from None
            if event is None:
                raise RuntimeError("Event stream ended before root turn completion")
            if isinstance(event, Exception):
                raise event
            yield event
    finally:
        stopped.set()


def run_turn(client, session_id, prompt, output, deadline, started):
    sessions = client.beta.agents.sessions
    previous_turns = {
        turn.id
        for turn in sessions.turns.list(
            session_id,
            limit=100,
            timeout=remaining(deadline),
        )
    }
    with sessions.events.stream(session_id, timeout=remaining(deadline)) as stream:
        sessions.events.create(
            session_id,
            events=[
                {
                    "type": "agent.session.input.message",
                    "input": [
                        {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
                    ],
                }
            ],
            timeout=remaining(deadline),
        )
        with (output / "events.jsonl").open("a") as evidence:
            for event in events_until(stream, deadline):
                raw = event.model_dump(mode="json")
                evidence.write(
                    redact(
                        json.dumps(
                            {
                                "received_at": now(),
                                "elapsed_seconds": time.monotonic() - started,
                                "event": raw,
                            }
                        )
                    )
                    + "\n"
                )
                evidence.flush()
                kind = raw.get("type", "")
                if kind in ("error", "agent.session.failed", "agent.session.environment.failed"):
                    raise RuntimeError(f"Agents API failure: {kind}")
                turn = raw.get("turn") or {}
                if (
                    kind
                    in (
                        "agent.session.turn.completed",
                        "agent.session.turn.failed",
                        "agent.session.turn.cancelled",
                    )
                    and turn.get("subagent_id") is None
                    and turn.get("id") not in previous_turns
                ):
                    if kind != "agent.session.turn.completed":
                        raise RuntimeError(f"Root turn failed: {kind}")
                    return turn.get("id")


def records(page, deadline):
    result = []
    remaining(deadline)
    for record in page:
        remaining(deadline)
        result.append(record.model_dump(mode="json"))
    return result


def collect_evidence(client, session_id, output, deadline):
    sessions = client.beta.agents.sessions
    turns = records(sessions.turns.list(session_id, order="asc", limit=100), deadline)
    items = records(sessions.items.list(session_id, order="asc", limit=100), deadline)
    children = records(sessions.subagents.list(session_id, order="asc", limit=100), deadline)
    dump(output / "subagents.json", children)
    for child in children:
        child_id = child["id"]
        turns.extend(
            records(
                sessions.subagents.turns.list(
                    child_id,
                    session_id=session_id,
                    order="asc",
                    limit=100,
                ),
                deadline,
            )
        )
        items.extend(
            records(
                sessions.subagents.items.list(
                    child_id,
                    session_id=session_id,
                    order="asc",
                    limit=100,
                ),
                deadline,
            )
        )
    # Some API versions include child turns in the parent list too.
    dump(output / "turns.json", list({turn["id"]: turn for turn in turns}.values()))
    dump(output / "items.json", list({item["id"]: item for item in items}.values()))


def collect_artifacts(sandbox, output, deadline):
    folder = output / "output"
    folder.mkdir(exist_ok=True)
    for name in ARTIFACTS:
        data = sandbox.read_file(
            f"{WORKSPACE}/output/{name}",
            timeout_seconds=remaining(deadline),
        ).result(timeout=remaining(deadline))
        text = data.decode("utf-8")
        if redact(text) != text:
            raise ValueError(f"Credential found in artifact {name}; refusing to save")
        (folder / name).write_text(text)


def cleanup(client, journal, output):
    errors = []
    auth = AUTH_MODES[journal.get("sandbox_auth", "coreweave")]

    def save():
        try:
            dump(output / "run.json", journal)
        except OSError as error:
            errors.append("Cleanup journal: " + type(error).__name__)

    session_id = journal.get("session_id")
    if session_id and not journal.get("session_deleted"):
        try:
            try:
                client.beta.agents.sessions.delete(session_id)
            except NotFoundError:
                pass
            journal["session_deleted"] = True
        except Exception as error:
            errors.append("API session: " + type(error).__name__)
        save()
    # The unique tag also finds a sandbox whose create response was lost.
    if journal.get("sandbox_stopped"):
        journal["cleanup_errors"] = errors
        save()
        return errors
    try:
        sandboxes = list(Sandbox.list(tags=[journal["sandbox_tag"]], auth=auth).result(timeout=30))
        if journal.get("sandbox_id") and not any(
            sb.sandbox_id == journal["sandbox_id"] for sb in sandboxes
        ):
            try:
                sandboxes.append(
                    Sandbox.from_id(journal["sandbox_id"], auth=auth).result(timeout=30)
                )
            except SandboxNotFoundError:
                pass
        for sandbox in sandboxes:
            sandbox.stop(missing_ok=True).result(timeout=60)
        journal["sandbox_stopped"] = True
    except Exception as error:
        errors.append("CWS sandbox: " + type(error).__name__)
    journal["cleanup_errors"] = errors
    save()
    return errors


def run(args, client):
    payloads = inputs()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 600
    journal = {
        "started_at": now(),
        "model": args.model,
        "status": "starting",
        "sandbox_tag": "openai-agents-recipe-" + uuid.uuid4().hex,
        "sandbox_lifetime_seconds": 1200,
        "runtime_deadline_seconds": 600,
        "sandbox_auth": getattr(args, "sandbox_auth", "coreweave"),
        "sandbox_mode": getattr(args, "sandbox_mode", None),
    }
    dump(output / "run.json", journal)
    sandbox = None
    try:
        session = client.beta.agents.sessions.create(
            agent={
                "model": args.model,
                "instructions": INSTRUCTIONS,
                "multi_agent": {"enabled": True, "max_concurrent_subagents": 3},
            },
            environment={"type": "self_hosted", "workspace_directory": WORKSPACE},
            timeout=remaining(deadline),
        )
        journal["session_id"] = session.id
        dump(output / "run.json", journal)
        env = environment(session)
        journal["environment_id"] = env.id
        sandbox = Sandbox.run(
            "sleep",
            "infinity",
            container_image="node:22-bookworm",
            max_lifetime_seconds=1200,
            request_timeout_seconds=30,
            auth=AUTH_MODES[journal["sandbox_auth"]],
            placement_mode=journal["sandbox_mode"],
            placement_spillover="strict",
            tags=[journal["sandbox_tag"]],
            resources=ResourceOptions(
                requests={"cpu": "2", "memory": "4Gi"}, limits={"cpu": "2", "memory": "4Gi"}
            ),
            environment_variables={
                "CODEX_API_KEY": os.environ["OPENAI_EXECUTOR_API_KEY"],
                "HOME": "/workspace/home",
            },
        )
        journal["sandbox_id"] = sandbox.sandbox_id
        dump(output / "run.json", journal)
        start_executor(sandbox, env, deadline)
        for name, data in payloads.items():
            sandbox.write_file(f"{WORKSPACE}/{name}", data).result(timeout=remaining(deadline))
        while True:
            status = client.beta.agents.environments.retrieve(
                env.id,
                timeout=remaining(deadline),
            ).status
            if status == "connected":
                break
            if status == "failed":
                raise RuntimeError("Executor failed to connect")
            time.sleep(min(1, remaining(deadline)))
        journal["status"] = "running"
        dump(output / "run.json", journal)
        print(f"Session: {session.id}\nSandbox: {sandbox.sandbox_id}", flush=True)
        journal["initial_turn_id"] = run_turn(
            client,
            session.id,
            payloads["TASK.md"].decode(),
            output,
            deadline,
            started,
        )
        collect_evidence(client, session.id, output, deadline)
        journal["followup_turn_id"] = run_turn(
            client,
            session.id,
            FOLLOWUP,
            output,
            deadline,
            started,
        )
        collect_evidence(client, session.id, output, deadline)
        collect_artifacts(sandbox, output, deadline)
        journal["status"] = "completed"
    except BaseException as error:
        journal["status"] = "failed"
        journal["error_type"] = type(error).__name__
        raise
    finally:
        journal["finished_at"] = now()
        if journal["status"] != "completed" and journal.get("session_id"):
            try:
                collect_evidence(client, journal["session_id"], output, time.monotonic() + 30)
            except Exception as error:
                journal["evidence_error_type"] = type(error).__name__
        if sandbox is not None:
            try:
                log = sandbox.read_file("/workspace/executor.log", timeout_seconds=5).result(
                    timeout=5,
                )
                (output / "executor.log").write_text(redact(log.decode("utf-8", errors="replace")))
            except Exception as error:
                journal["executor_log_error_type"] = type(error).__name__
        errors = cleanup(client, journal, output)
        if errors:
            print("Cleanup incomplete. Re-run the cleanup command with this output directory.")
    if errors:
        raise RuntimeError("Resource cleanup incomplete")
    print(f"Artifacts and API evidence saved in {output}. Run verify.py before claiming success.")


def interrupt(*_):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "cleanup"):
        sub = commands.add_parser(name)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument(
            "--env-file",
            type=Path,
            help="Load variables from this file, overriding exported values",
        )
        if name == "run":
            sub.add_argument("--model", default="gpt-6-astra")
            sub.add_argument("--sandbox-auth", choices=AUTH_MODES, default="coreweave")
            sub.add_argument("--sandbox-mode", choices=("serverless", "cks"))
    args = parser.parse_args(argv)
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("--env-file does not exist")
        load_dotenv(args.env_file, override=True)
    if args.command == "run":
        mode = args.sandbox_auth
    else:
        journal = json.loads((args.output_dir / "run.json").read_text())
        mode = journal.get("sandbox_auth", "coreweave")
        if mode not in AUTH_MODES:
            parser.error("Unknown sandbox authentication mode in run.json")
    required = [AUTH_KEYS[mode], "OPENAI_API_KEY"]
    if args.command == "run":
        required.append("OPENAI_EXECUTOR_API_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        parser.error("Missing environment variables: " + ", ".join(missing))
    signal.signal(signal.SIGTERM, interrupt)
    with OpenAI(max_retries=0, timeout=30) as client:
        if args.command == "run":
            run(args, client)
        else:
            if cleanup(client, journal, args.output_dir):
                raise SystemExit("Cleanup incomplete; see run.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
