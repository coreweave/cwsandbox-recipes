"""Run upstream mini-swe-agent locally with shell actions in the task sandbox."""

import asyncio
import os

import yaml
from cwsandbox.exceptions import SandboxTimeoutError
from minisweagent import __version__, package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import InterruptAgentFlow
from minisweagent.models.litellm_model import LitellmModel
from pier.agents.base import BaseAgent
from pier.agents.installed.mini_swe_agent import convert_and_save_trajectory


class RemoteShell(LocalEnvironment):
    def __init__(self, environment, loop):
        super().__init__(cwd="/app", timeout=120)
        self.environment = environment
        self.loop = loop

    def execute(self, action, cwd="", *, timeout=None):
        future = asyncio.run_coroutine_threadsafe(
            self.environment.exec(
                action["command"], cwd=cwd or "/app", timeout_sec=timeout or 120
            ),
            self.loop,
        )
        try:
            result = future.result()
        except SandboxTimeoutError:
            # A slow shell command is an agent observation, not a failed trial.
            # Infrastructure/authentication failures still propagate to Pier.
            return {
                "output": "",
                "returncode": -1,
                "exception_info": f"Command timed out after {timeout or 120} seconds.",
            }
        output = {
            "output": (result.stdout or "") + (result.stderr or ""),
            "returncode": result.return_code,
            "exception_info": "",
        }
        self._check_finished(output)
        return output

    def get_template_vars(self, **kwargs):
        return dict(system="Linux", release="", version="", machine="x86_64", **kwargs)


class HostMiniSweAgent(BaseAgent):
    SUPPORTS_ATIF = True

    def __init__(self, *args, step_limit=160, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_limit = step_limit

    @staticmethod
    def name():
        return "mini-swe-agent-host"

    def version(self):
        return __version__

    async def setup(self, environment):
        result = await environment.exec(
            "git config --global --add safe.directory /app && "
            "git config --global user.email agent@example.invalid && "
            "git config --global user.name 'Benchmark agent'",
            cwd="/app",
        )
        if result.return_code:
            raise RuntimeError("Could not configure task Git identity")

    async def run(self, instruction, environment, context):
        # LiteLLM reads OPENAI_API_KEY from process memory, not serialized config.
        os.environ["OPENAI_API_KEY"] = os.environ["WANDB_API_KEY"]
        config = yaml.safe_load((package_dir / "config/mini.yaml").read_text())
        model = LitellmModel(
            model_name=f"openai/{self.model_name}",
            model_kwargs={
                "api_base": "https://api.inference.wandb.ai/v1",
                "max_tokens": 8192,
                "timeout": 180,
                "num_retries": 2,
            },
            cost_tracking="ignore_errors",
            observation_template=config["model"]["observation_template"],
        )
        path = self.logs_dir / "mini.trajectory.json"
        agent = DefaultAgent(
            model,
            RemoteShell(environment, asyncio.get_running_loop()),
            system_template=config["agent"]["system_template"],
            instance_template=config["agent"]["instance_template"],
            step_limit=self.step_limit,
            cost_limit=0,
            output_path=path,
        )
        agent.extra_template_vars = {"task": instruction}
        agent.add_messages(
            model.format_message(
                role="system",
                content=agent._render_template(agent.config.system_template),
            ),
            model.format_message(
                role="user",
                content=agent._render_template(agent.config.instance_template),
            ),
        )
        try:
            while not agent.messages or agent.messages[-1].get("role") != "exit":
                try:
                    await asyncio.to_thread(agent.step)
                except InterruptAgentFlow as exc:
                    agent.add_messages(*exc.messages)
                finally:
                    agent.save(path)
                    context.n_agent_steps = agent.n_calls
                    self.logger.info("Completed model step %d", agent.n_calls)
        finally:
            if path.exists():
                trajectory = convert_and_save_trajectory(
                    path, self.logs_dir / "trajectory.json", environment.session_id
                )
                if trajectory.final_metrics:
                    context.n_input_tokens = (
                        trajectory.final_metrics.total_prompt_tokens
                    )
                    context.n_output_tokens = (
                        trajectory.final_metrics.total_completion_tokens
                    )
                    context.n_cache_tokens = (
                        trajectory.final_metrics.total_cached_tokens
                    )
                context.metadata = {
                    "exit_status": agent.messages[-1]
                    .get("extra", {})
                    .get("exit_status"),
                    "cost_note": "Use W&B billing for actual inference charges.",
                }
