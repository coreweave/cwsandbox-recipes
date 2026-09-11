"""Tests for the sandbox pool and the veRL tool that leases from it.

No cwsandbox and no network. A fake backend hands out in-process `EpisodeState`
objects, and a fake transport routes HTTP calls straight into the real
`env_server._dispatch`. So the routing, reward and reset logic under test are the
shipped ones -- only sandbox provisioning is faked.
"""

import asyncio
from typing import Any, Dict, Optional, Tuple

import pytest
from helpers import ScriptedSimulator

from verl_taubench._verl_compat import VERL_AVAILABLE
from verl_taubench.sandbox.client import EnvServerError, SandboxEnvClient
from verl_taubench.sandbox.env_server import EpisodeError, EpisodeState, _dispatch
from verl_taubench.sandbox.pool import SandboxHandle, SandboxPool
from verl_taubench.sandbox.simulator import STOP_TOKEN
from verl_taubench.tools.sandbox_taubench_tool import SandboxTauBenchTool


class InProcessTransport:
    """Routes requests to real EpisodeState objects registered by base_url."""

    def __init__(self) -> None:
        self.servers: Dict[str, EpisodeState] = {}
        self.dead: set = set()
        self.requests: list = []
        self.closed = False

    def register(self, base_url: str, state: EpisodeState) -> None:
        self.servers[base_url] = state

    def kill(self, base_url: str) -> None:
        """Simulate a crashed sandbox: connections fail."""
        self.dead.add(base_url)

    async def request(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        base_url, _, leaf = url.rpartition("/")
        path = "/" + leaf
        self.requests.append((method, base_url, path, json_body))
        if base_url in self.dead:
            raise ConnectionError(f"sandbox {base_url} is unreachable")
        state = self.servers.get(base_url)
        if state is None:
            return 404, {"error": "no such sandbox"}
        try:
            return 200, _dispatch(state, path, json_body or {})
        except EpisodeError as exc:
            return exc.status, {"error": str(exc)}

    async def aclose(self) -> None:
        self.closed = True


class FakeBackend:
    """Provisions in-process env servers instead of real sandboxes."""

    def __init__(self, transport: InProcessTransport, *, domain="retail", task_split="test"):
        self.transport = transport
        self.domain = domain
        self.task_split = task_split
        self.provisioned = 0
        self.destroyed = 0
        self.fail_next = 0

    async def provision(self) -> SandboxHandle:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("provisioning failed (injected)")
        self.provisioned += 1
        sandbox_id = f"sbx-{self.provisioned}"
        base_url = f"http://fake/{sandbox_id}"
        self.transport.register(base_url, EpisodeState(self.domain, self.task_split))
        return SandboxHandle(sandbox_id=sandbox_id, base_url=base_url, token="tok")

    async def destroy(self, handle: SandboxHandle) -> None:
        self.destroyed += 1


def _pool(size: int = 2, **kw) -> Tuple[SandboxPool, FakeBackend, InProcessTransport]:
    transport = InProcessTransport()
    backend = FakeBackend(transport)
    # Short acquire timeout so a leaked semaphore slot fails the test quickly
    # instead of hanging the suite forever.
    kw.setdefault("acquire_timeout_s", 2.0)
    return SandboxPool(backend, transport, size=size, **kw), backend, transport


# -- pool semantics --------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_loads_task_and_release_returns_capacity() -> None:
    pool, backend, _ = _pool(size=1)
    lease = await pool.acquire(task_index=3, episode_id="ep1")

    assert lease.task_index == 3
    assert lease.metadata["task_index"] == 3
    assert lease.metadata["instruction"]
    assert pool.live_count == 1

    await pool.release(lease)
    assert pool.idle_count == 1

    # The same sandbox is reused rather than reprovisioned.
    again = await pool.acquire(task_index=5, episode_id="ep2")
    assert again.handle.sandbox_id == lease.handle.sandbox_id
    assert backend.provisioned == 1
    assert again.metadata["task_index"] == 5, "reused sandbox must be reset to the new task"
    await pool.release(again)
    await pool.aclose()


@pytest.mark.asyncio
async def test_size_is_a_hard_ceiling_on_concurrent_leases() -> None:
    pool, backend, _ = _pool(size=2)
    a = await pool.acquire(0, "a")
    b = await pool.acquire(1, "b")
    assert backend.provisioned == 2

    # A third acquire must block until one is released.
    third = asyncio.create_task(pool.acquire(2, "c"))
    await asyncio.sleep(0.05)
    assert not third.done(), "pool exceeded its size ceiling"
    assert backend.provisioned == 2

    await pool.release(a)
    lease_c = await asyncio.wait_for(third, timeout=2)
    assert backend.provisioned == 2, "should reuse the released sandbox, not provision"

    await pool.release(b)
    await pool.release(lease_c)
    await pool.aclose()


@pytest.mark.asyncio
async def test_no_two_episodes_share_a_sandbox_concurrently() -> None:
    pool, _, _ = _pool(size=3)
    leases = [await pool.acquire(i, f"ep{i}") for i in range(3)]
    ids = {lease.handle.sandbox_id for lease in leases}
    assert len(ids) == 3, "the same sandbox was leased to two episodes"
    for lease in leases:
        await pool.release(lease)
    await pool.aclose()


@pytest.mark.asyncio
async def test_dead_sandbox_is_replaced_not_handed_out() -> None:
    pool, backend, transport = _pool(size=1)
    lease = await pool.acquire(0, "ep1")
    await pool.release(lease)

    transport.kill(lease.handle.base_url)

    replacement = await pool.acquire(1, "ep2")
    assert replacement.handle.sandbox_id != lease.handle.sandbox_id
    assert backend.destroyed >= 1
    assert pool.stats["replacements"] >= 1
    assert replacement.metadata["task_index"] == 1
    await pool.release(replacement)
    await pool.aclose()


@pytest.mark.asyncio
async def test_release_always_returns_capacity_even_when_unhealthy() -> None:
    """An unhealthy sandbox must be destroyed, not leaked -- capacity must recover."""
    pool, backend, _ = _pool(size=1)
    lease = await pool.acquire(0, "ep1")
    await pool.release(lease, healthy=False)

    assert backend.destroyed == 1
    assert pool.idle_count == 0

    # Capacity recovered: a fresh acquire succeeds rather than hanging.
    nxt = await asyncio.wait_for(pool.acquire(1, "ep2"), timeout=2)
    assert nxt.handle.sandbox_id != lease.handle.sandbox_id
    await pool.release(nxt)
    await pool.aclose()


@pytest.mark.asyncio
async def test_lease_contextmanager_marks_unhealthy_on_exception() -> None:
    pool, backend, _ = _pool(size=1)

    with pytest.raises(ValueError):
        async with pool.lease(0, "ep1"):
            raise ValueError("episode blew up")

    assert backend.destroyed == 1, "a crashed episode must not return its sandbox to the pool"
    assert pool.idle_count == 0
    # Capacity still recovered.
    lease = await asyncio.wait_for(pool.acquire(0, "ep2"), timeout=2)
    await pool.release(lease)
    await pool.aclose()


@pytest.mark.asyncio
async def test_acquire_retries_transient_provisioning_failure() -> None:
    pool, backend, _ = _pool(size=1, max_provision_retries=2)
    backend.fail_next = 2
    lease = await pool.acquire(0, "ep1")
    assert backend.provisioned == 1
    await pool.release(lease)
    await pool.aclose()


@pytest.mark.asyncio
async def test_permanent_provision_failure_does_not_leak_a_slot() -> None:
    """A failed acquire must give its slot back, or the pool deadlocks forever.

    `acquire` takes the semaphore before provisioning, so every failure path
    between there and the return has to release it.
    """
    pool, backend, _ = _pool(size=1, max_provision_retries=1)
    backend.fail_next = 99  # never recovers

    with pytest.raises(RuntimeError):
        await pool.acquire(0, "doomed")

    # If the slot leaked, this blocks until the timeout instead of succeeding.
    backend.fail_next = 0
    lease = await asyncio.wait_for(pool.acquire(0, "recovered"), timeout=2)
    await pool.release(lease)
    await pool.aclose()


@pytest.mark.asyncio
async def test_permanent_reset_failure_does_not_leak_a_slot() -> None:
    """Same invariant, but for a sandbox that provisions yet never resets."""
    pool, backend, transport = _pool(size=1, max_provision_retries=1)

    original = backend.provision

    async def provision_dead():
        handle = await original()
        transport.kill(handle.base_url)  # provisions fine, unreachable forever
        return handle

    backend.provision = provision_dead  # type: ignore[method-assign]

    with pytest.raises(Exception):
        await pool.acquire(0, "doomed")

    backend.provision = original  # type: ignore[method-assign]
    lease = await asyncio.wait_for(pool.acquire(0, "recovered"), timeout=2)
    await pool.release(lease)
    await pool.aclose()


@pytest.mark.asyncio
async def test_prewarm_fills_the_pool_and_aclose_destroys_everything() -> None:
    pool, backend, transport = _pool(size=3)
    await pool.prewarm()
    assert backend.provisioned == 3
    assert pool.idle_count == 3

    await pool.aclose()
    assert backend.destroyed == 3
    assert pool.live_count == 0
    assert transport.closed is True


@pytest.mark.asyncio
async def test_client_surfaces_server_errors() -> None:
    transport = InProcessTransport()
    transport.register("http://fake/s1", EpisodeState("retail", "test"))
    client = SandboxEnvClient("http://fake/s1", transport, token="tok")

    # Stepping before reset is a 409 from the real server.
    with pytest.raises(EnvServerError) as exc:
        await client.step("respond", {"content": "hi"})
    assert exc.value.status == 409


# -- tool integration ------------------------------------------------------


async def _make_tool(replies, size: int = 2):
    pool, backend, transport = _pool(size=size)
    simulator = ScriptedSimulator(replies=replies)
    tool = SandboxTauBenchTool(config={}, pool=pool, simulator=simulator)
    return tool, pool, backend, simulator


@pytest.mark.asyncio
async def test_tool_routes_respond_to_simulator_and_records_in_sandbox() -> None:
    tool, pool, _, simulator = await _make_tool(replies=["tell me more"])
    instance_id, first = await tool.create(domain="retail", task_split="test", task_index=2)

    assert first.text == simulator.opening, "the opening user turn comes from the simulator"

    response, reward, metrics = await tool.execute(
        instance_id, {"action_name": "respond", "action_kwargs": {"content": "You have 10."}}
    )

    assert response.text == "tell me more", "observation must be the simulator's reply"
    assert metrics["source"] == "user"
    assert reward == 0.0

    # And it was recorded in the sandbox, which is what reward scoring needs.
    lease = tool.instance_registry[instance_id]["lease"]
    state = pool.transport.servers[lease.handle.base_url]  # type: ignore[attr-defined]
    recorded = [a for a in state.env.actions if a.name == "respond"]
    assert [a.kwargs["content"] for a in recorded] == ["You have 10."]

    await tool.release(instance_id)
    await pool.aclose()


@pytest.mark.asyncio
async def test_tool_routes_tool_calls_to_the_sandbox() -> None:
    tool, pool, _, _ = await _make_tool(replies=["ok"])
    instance_id, _ = await tool.create(domain="retail", task_split="test", task_index=0)

    response, _, metrics = await tool.execute(
        instance_id, {"action_name": "list_all_product_types", "action_kwargs": {}}
    )

    assert response.text, "tool output must come back as the observation"
    assert metrics["source"] == "list_all_product_types"
    assert metrics["done"] is False

    await tool.release(instance_id)
    await pool.aclose()


@pytest.mark.asyncio
async def test_stop_token_ends_episode_and_triggers_reward() -> None:
    tool, pool, _, _ = await _make_tool(replies=[STOP_TOKEN])
    instance_id, _ = await tool.create(domain="retail", task_split="test", task_index=2)

    _, reward, metrics = await tool.execute(
        instance_id, {"action_name": "respond", "action_kwargs": {"content": "You have 10."}}
    )

    assert metrics["done"] is True
    assert isinstance(reward, float)
    assert tool.instance_registry[instance_id]["done"] is True

    await tool.release(instance_id)
    await pool.aclose()


@pytest.mark.asyncio
async def test_missing_respond_content_is_a_recoverable_error() -> None:
    tool, pool, _, _ = await _make_tool(replies=["ok"])
    instance_id, _ = await tool.create(domain="retail", task_split="test", task_index=0)

    response, reward, metrics = await tool.execute(
        instance_id, {"action_name": "respond", "action_kwargs": {}}
    )

    assert metrics["error"] == "missing_respond_content"
    assert "must include" in response.text
    assert reward == 0.0
    # The episode is still usable.
    assert tool.instance_registry[instance_id]["done"] is False

    await tool.release(instance_id)
    await pool.aclose()


@pytest.mark.asyncio
async def test_release_returns_the_lease_and_clears_simulator_state() -> None:
    tool, pool, _, simulator = await _make_tool(replies=["ok"], size=1)
    instance_id, _ = await tool.create(domain="retail", task_split="test", task_index=0)
    assert pool.idle_count == 0

    await tool.release(instance_id)

    assert instance_id not in tool.instance_registry
    assert pool.idle_count == 1, "lease was not returned to the pool"
    assert {c["event"] for c in simulator.calls if c["episode_id"] == instance_id} >= {"finish"}
    await pool.aclose()


@pytest.mark.asyncio
async def test_dataset_domain_mismatch_is_rejected() -> None:
    """A pool is pinned to one domain; a mismatched row must fail loudly.

    Otherwise an `airline` dataset row against a retail pool trains on retail
    tasks with a retail system prompt, silently.
    """
    pool, _, _ = _pool(size=1)
    tool = SandboxTauBenchTool(
        config={"backend": {"domain": "retail", "task_split": "test"}},
        pool=pool,
        simulator=ScriptedSimulator(replies=["ok"]),
    )

    with pytest.raises(ValueError, match="does not match the sandbox pool"):
        await tool.create(domain="airline", task_split="test", task_index=0)

    with pytest.raises(ValueError, match="task_split"):
        await tool.create(domain="retail", task_split="train", task_index=0)

    # A matching row still works, and no lease leaked from the rejections.
    instance_id, _ = await tool.create(domain="retail", task_split="test", task_index=0)
    await tool.release(instance_id)
    await pool.aclose()


@pytest.mark.asyncio
async def test_create_requires_task_index_and_does_not_leak_a_lease() -> None:
    tool, pool, _, _ = await _make_tool(replies=["ok"], size=1)

    with pytest.raises(ValueError):
        await tool.create(domain="retail", task_split="test")

    # No lease was taken, so full capacity remains.
    lease = await asyncio.wait_for(pool.acquire(0, "ep"), timeout=2)
    await pool.release(lease)
    await pool.aclose()


@pytest.mark.asyncio
@pytest.mark.skipif(
    VERL_AVAILABLE,
    reason="stub-mode loop test: the fake tokenizer/config cannot satisfy the "
    "real verl AgentLoopBase; runs where verl is not installed",
)
async def test_sandbox_tool_drives_the_real_agent_loop() -> None:
    """The sandboxed tool must be a genuine drop-in for the in-process one.

    Constructs the production `TauBenchAgentLoop` with a `SandboxTauBenchTool`
    and runs a full episode. Guards two couplings that an `isinstance` check and
    a `state["env"]` lookup would silently break.
    """
    from verl_taubench.agent.taubench_loop import TauBenchAgentLoop

    response_text = (
        '<tool_call>{"action_name": "respond", '
        '"action_kwargs": {"content": "You have 10."}}</tool_call>'
    )

    class Tok:
        chat_template = True

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw):
            return "\n".join(f"<{m['role']}>{m.get('content','')}</{m['role']}>" for m in msgs)

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(int(i)) for i in ids)

    class Server:
        async def generate(self, request_id, prompt_ids, sampling_params):
            return [ord(c) for c in response_text]

    pool, _, _ = _pool(size=1)
    simulator = ScriptedSimulator(replies=[STOP_TOKEN])
    tool = SandboxTauBenchTool(config={}, pool=pool, simulator=simulator)

    loop = TauBenchAgentLoop(
        trainer_config=None,
        server_manager=Server(),
        tokenizer=Tok(),
        processor=None,
        dataset_cls=None,
        data_config=None,
        tools=[tool],
        max_turns=3,
    )
    assert loop.step_tool is tool, "the loop rejected the sandboxed tool"

    output = await loop.run(
        {}, extra_info={"domain": "retail", "task_split": "test", "task_index": 2}
    )

    # No raw_prompt was supplied, so the loop had to build the system prompt from
    # the sandbox /spec rather than from a local env object.
    prompt_text = Tok().decode(output.prompt_ids)
    assert "Available tools:" in prompt_text and "<tool_call>" in prompt_text

    assert len(output.response_ids) == len(output.response_mask)
    assert output.num_turns >= 1
    assert isinstance(output.reward_score, float)
    # The lease must have been returned by the loop's finally: release().
    assert pool.idle_count == 1
    await pool.aclose()


@pytest.mark.asyncio
async def test_sandbox_prompt_matches_in_process_prompt_exactly() -> None:
    """The two tracks must produce byte-identical system prompts."""
    from verl_taubench.envs import taubench_env

    pool, _, _ = _pool(size=1)
    tool = SandboxTauBenchTool(config={}, pool=pool, simulator=ScriptedSimulator(replies=["ok"]))
    instance_id, _ = await tool.create(domain="retail", task_split="test", task_index=0)
    sandbox_prompt = tool.instance_registry[instance_id]["system_prompt"]

    local_env = taubench_env.make_env(
        domain="retail", task_split="test", task_index=0, user_strategy="human"
    )
    local_prompt = taubench_env.build_system_prompt(local_env)

    assert sandbox_prompt == local_prompt, "sandboxed and in-process prompts diverged"

    await tool.release(instance_id)
    await pool.aclose()


@pytest.mark.asyncio
async def test_correct_episode_scores_one_and_wrong_episode_scores_zero() -> None:
    """The reward channel must actually carry signal, not just a float.

    Without this, silently returning 0.0 everywhere keeps the suite green while
    GRPO trains on a constant-zero advantage and learns nothing.

    Replays the task's ground-truth actions (so the DB diff matches the goal
    state) and says the required output, which is the only way τ-bench awards 1.0.
    """
    from tau_bench.types import RESPOND_ACTION_NAME

    from verl_taubench.envs import taubench_env

    task_index = 2
    reference = taubench_env.make_env(
        domain="retail", task_split="test", task_index=task_index, user_strategy="human"
    )
    gt_actions = [a for a in reference.task.actions if a.name != RESPOND_ACTION_NAME]
    required_output = reference.task.outputs[0]
    assert required_output == "10", "fixture drifted"

    # --- correct episode -------------------------------------------------
    pool, _, _ = _pool(size=1)
    tool = SandboxTauBenchTool(
        config={}, pool=pool, simulator=ScriptedSimulator(replies=[STOP_TOKEN])
    )
    instance_id, _ = await tool.create(
        domain="retail", task_split="test", task_index=task_index
    )
    for action in gt_actions:
        await tool.execute(
            instance_id, {"action_name": action.name, "action_kwargs": dict(action.kwargs)}
        )
    _, reward, metrics = await tool.execute(
        instance_id,
        {"action_name": "respond", "action_kwargs": {"content": f"You have {required_output}."}},
    )
    assert metrics["done"] is True
    assert reward == 1.0, f"a correct episode must score 1.0, got {reward}"
    assert await tool.calc_reward(instance_id) == 1.0
    await tool.release(instance_id)
    await pool.aclose()

    # --- wrong episode ---------------------------------------------------
    pool2, _, _ = _pool(size=1)
    tool2 = SandboxTauBenchTool(
        config={}, pool=pool2, simulator=ScriptedSimulator(replies=[STOP_TOKEN])
    )
    instance_id2, _ = await tool2.create(
        domain="retail", task_split="test", task_index=task_index
    )
    _, wrong_reward, _ = await tool2.execute(
        instance_id2, {"action_name": "respond", "action_kwargs": {"content": "I don't know."}}
    )
    assert wrong_reward == 0.0, "an episode that never did the work must score 0.0"
    await tool2.release(instance_id2)
    await pool2.aclose()


@pytest.mark.asyncio
@pytest.mark.skipif(
    VERL_AVAILABLE,
    reason="stub-mode loop test: the fake tokenizer/config cannot satisfy the "
    "real verl AgentLoopBase; runs where verl is not installed",
)
async def test_sandbox_tool_owns_turn_zero_and_does_not_leak_the_instruction() -> None:
    """The dataset's raw_prompt carries the simulator's private brief.

    `task.instruction` is what the user simulator is told to reveal gradually.
    Handing it to the policy as turn 0 both defeats the simulator and wastes the
    opening completion we already paid for.
    """
    from verl_taubench.agent.taubench_loop import TauBenchAgentLoop

    class Tok:
        chat_template = True

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw):
            return "\n".join(f"<{m['role']}>{m.get('content','')}</{m['role']}>" for m in msgs)

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(int(i)) for i in ids)

    class Server:
        async def generate(self, request_id, prompt_ids, sampling_params):
            return [ord(c) for c in "hello"]

    pool, _, _ = _pool(size=1)
    simulator = ScriptedSimulator(replies=[STOP_TOKEN], opening="Hi, I have a question.")
    tool = SandboxTauBenchTool(config={}, pool=pool, simulator=simulator)
    assert tool.owns_initial_turn is True

    loop = TauBenchAgentLoop(
        trainer_config=None,
        server_manager=Server(),
        tokenizer=Tok(),
        processor=None,
        dataset_cls=None,
        data_config=None,
        tools=[tool],
        max_turns=1,
    )

    # The dataset supplies a raw_prompt containing the hidden instruction.
    output = await loop.run(
        {},
        extra_info={"domain": "retail", "task_split": "test", "task_index": 2},
        raw_prompt=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "SECRET_INSTRUCTION_MARKER"},
        ],
    )

    prompt_text = Tok().decode(output.prompt_ids)
    assert "SECRET_INSTRUCTION_MARKER" not in prompt_text, (
        "the simulator's private brief leaked into the policy prompt"
    )
    assert "Hi, I have a question." in prompt_text, (
        "turn 0 must be the simulator's opening utterance"
    )
    await pool.aclose()


@pytest.mark.asyncio
async def test_spec_is_fetched_once_per_domain_not_once_per_episode() -> None:
    """The ~17 KB wiki/rules/schema block must not be refetched every episode."""
    pool, _, transport = _pool(size=1)
    tool = SandboxTauBenchTool(config={}, pool=pool, simulator=ScriptedSimulator(replies=["ok"]))

    for i in range(3):
        instance_id, _ = await tool.create(domain="retail", task_split="test", task_index=i)
        await tool.release(instance_id)

    spec_calls = [r for r in transport.requests if r[2] == "/spec"]
    assert len(spec_calls) == 1, f"expected 1 /spec fetch across 3 episodes, got {len(spec_calls)}"
    await pool.aclose()


@pytest.mark.asyncio
async def test_simulator_failure_during_create_does_not_leak_a_lease() -> None:
    """A lease must never outlive a failed create -- that would exhaust the pool."""

    class ExplodingSimulator(ScriptedSimulator):
        async def start(self, episode_id: str, instruction: str) -> str:
            raise RuntimeError("simulator down")

    pool, _, _ = _pool(size=1)
    tool = SandboxTauBenchTool(config={}, pool=pool, simulator=ExplodingSimulator())

    with pytest.raises(RuntimeError, match="simulator down"):
        await tool.create(domain="retail", task_split="test", task_index=0)

    # Capacity recovered rather than being held by the dead episode.
    lease = await asyncio.wait_for(pool.acquire(0, "ep"), timeout=2)
    await pool.release(lease)
    await pool.aclose()
