"""Run AutomationBench's native evaluator as a Harbor agent, without changing scoring."""

import json
import os
import re
import shlex

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import ApiRateLimitError, ApiUsageLimitError

REVISION = "4a8e1061254004d9dac807054eed33fad7d1ff14"


def terminal_quota_error(log):
    terminal = "\n".join(
        line
        for line in log.splitlines()
        if "Aborted rollout" in line
        or re.match(
            r"(?:openai\.)?(?:RateLimitError|ApiUsageLimitError|QuotaExceededError):",
            line,
        )
    )
    if re.search(
        r"insufficient_quota|usage limit|quota exceeded|credits.*exhaust",
        terminal,
        re.IGNORECASE,
    ):
        return ApiUsageLimitError
    if re.search(r"RateLimitError|Error code: 429|Concurrency limit", terminal):
        return ApiRateLimitError
    return None


class NativeAutomationBenchAgent(BaseAgent):
    @staticmethod
    def name():
        return "automationbench-native"

    def version(self):
        return REVISION

    async def setup(self, environment):
        command = (
            "apt-get update -qq && apt-get install -y -qq git ca-certificates "
            "&& pip install --quiet uv==0.8.22 "
            "&& git clone -q https://github.com/zapier/AutomationBench.git /workspace/AutomationBench "
            f"&& cd /workspace/AutomationBench && git checkout -q {REVISION} "
            "&& uv sync --frozen --no-dev"
        )
        result = await environment.exec(command, timeout_sec=600)
        (self.logs_dir / "setup.log").write_text(
            (result.stdout or "") + (result.stderr or "")
        )
        if result.return_code:
            raise RuntimeError("AutomationBench setup failed; see setup.log")

    async def run(self, instruction, environment, context):
        spec = json.loads(instruction)
        name = spec["task"]
        if not re.fullmatch(r"[a-z]+\.[a-z0-9_]+", name):
            raise ValueError("Invalid AutomationBench task name")
        model_env = {
            key: os.environ[key]
            for key in ["MODEL_API_KEY", "MODEL_NAME", "MODEL_BASE_URL"]
        }
        if project := os.environ.get("MODEL_PROJECT"):
            model_env["MODEL_PROJECT"] = project
        command = [
            "uv",
            "run",
            "--frozen",
            "--no-dev",
            "auto-bench",
            "--model",
            model_env["MODEL_NAME"],
            "--base-url",
            model_env["MODEL_BASE_URL"],
            "--api-key-var",
            "MODEL_API_KEY",
            "--api",
            "chat_completions",
            "--toolset",
            "api",
            "--domains",
            name.split(".")[0],
            "--tasks",
            name,
            "--num-examples",
            "-1",
            "--max-concurrent",
            "1",
            "--max-steps",
            "50",
            "--no-ensure-complete",
            "--export-json",
            "/logs/agent/automationbench.json",
        ]
        if project:
            command += ["--headers", "OpenAI-Project=" + project]
        (self.logs_dir / "command.json").write_text(json.dumps(command, indent=2))
        result = await environment.exec(
            shlex.join(command) + " > /logs/agent/eval.log 2>&1",
            cwd="/workspace/AutomationBench",
            timeout_sec=2400,
        )
        await environment.download_file(
            "/logs/agent/eval.log", self.logs_dir / "eval.log"
        )
        log = (self.logs_dir / "eval.log").read_text()
        # Only terminal rollout errors trigger trial retries. Transient warnings
        # already handled successfully inside the native client do not rerun a task.
        if error_type := terminal_quota_error(log):
            raise error_type(
                "Inference quota/rate limit prevented task completion; see archived eval.log"
            )
        await environment.download_file(
            "/logs/agent/automationbench.json", self.logs_dir / "automationbench.json"
        )
        data = json.loads((self.logs_dir / "automationbench.json").read_text())
        if result.return_code or [t["name"] for t in data.get("tasks", [])] != [name]:
            raise RuntimeError(
                "Native evaluator did not export exactly the requested task"
            )
