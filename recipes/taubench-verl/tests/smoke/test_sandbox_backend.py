"""Backend behavior against a fake SDK, plus installed-SDK signature checks.

The fake deliberately mirrors the real API's sharp edges:
  * `Session.sandbox(...)` returns unstarted; `wait()` starts and blocks.
  * `exec()` returns a Process whose `.result()` yields `.returncode`
    (NOT `.exit_code`).
  * `write_file()` returns an OperationRef requiring `.result()`.
  * Reachability is `service_urls` of `(port, name, url)` tuples.
  * Ports are declared with `services=[Service(...)]`, not `NetworkOptions`.
"""

import signal
import sys
import threading
import types
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import pytest

from verl_taubench.sandbox.cleanup import cleanup_tag
from verl_taubench.sandbox.pool import CwSandboxBackend


@pytest.mark.parametrize("auth", ["wandb", "coreweave"])
def test_provision_arguments_match_installed_sdk_without_starting_a_sandbox(monkeypatch, auth):
    sdk = pytest.importorskip("cwsandbox")
    import wandb

    monkeypatch.setattr(wandb, "run", None)
    monkeypatch.setenv("CWSANDBOX_SERVERLESS_AUTH", auth)

    class StartIntercepted(Exception):
        pass

    def intercept_start(self, timeout=None):
        raise StartIntercepted

    async def no_remote_stop(self, **kwargs):
        pass

    monkeypatch.setattr(sdk.Sandbox, "wait", intercept_start)
    monkeypatch.setattr(sdk.Sandbox, "_stop_async", no_remote_stop)
    backend = _backend(placement_mode="serverless")
    try:
        with pytest.raises(StartIntercepted):
            backend._provision_blocking()
        assert backend._session.get_metrics()["cwsandbox/sandboxes_created"] == 1
    finally:
        if backend._session is not None:
            backend._session.close().result()


class FakeProcessResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRef:
    def __init__(self, value: Any = None, raises: Optional[Exception] = None):
        self._value = value
        self._raises = raises

    def result(self, timeout: Optional[float] = None) -> Any:
        if self._raises is not None:
            raise self._raises
        return self._value


class FakeSandbox:
    """Records every call so tests can assert ordering and arguments."""

    instances: List["FakeSandbox"] = []
    listed: List["FakeSandbox"] = []
    list_tags: List[List[str]] = []
    list_auth: List[Any] = []

    # Injected failure switches.
    fail_list = False
    fail_wait = False
    fail_service_address = False
    fail_install = False
    fail_write = False
    fail_launch = False
    fail_run = False

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.sandbox_id = f"fake-{len(type(self).instances) + 1}"
        self.calls: List[str] = []
        self.execs: List[Dict[str, Any]] = []
        self.written: Dict[str, bytes] = {}
        self.stopped = False
        self._address = None if type(self).fail_service_address else "10.0.0.7:8080"
        self._service_urls = (
            ()
            if type(self).fail_service_address
            else ((8080, "env", "https://10.0.0.7"),)
        )
        type(self).instances.append(self)

    @classmethod
    def run(cls, *args: str, **kwargs: Any) -> "FakeSandbox":
        if kwargs.get("profile_ids") is not None or kwargs.get("profile_names") is not None:
            raise TypeError("profile_ids/profile_names were removed in cwsandbox 1.x")
        if cls.fail_run:
            raise RuntimeError("cwsandbox rejected Sandbox.run")
        sandbox = cls(**kwargs)
        sandbox.command = list(args)
        sandbox.calls.append("run")
        return sandbox

    @classmethod
    def list(cls, *, tags: List[str], auth: Any = None) -> FakeRef:
        cls.list_tags.append(list(tags))
        cls.list_auth.append(auth)
        if cls.fail_list:
            raise RuntimeError("cwsandbox rejected Sandbox.list")
        # The real API excludes stopped sandboxes (show_terminated=False).
        return FakeRef([s for s in cls.listed if not s.stopped])

    @property
    def service_address(self) -> Optional[str]:
        return self._address

    @property
    def service_urls(self) -> tuple:
        return self._service_urls

    def wait(self, timeout: Optional[float] = None) -> "FakeSandbox":
        self.calls.append("wait")
        if type(self).fail_wait:
            raise TimeoutError("sandbox did not reach RUNNING")
        return self

    def exec(self, command, timeout_seconds: Optional[float] = None, **kw: Any) -> FakeRef:
        self.calls.append("exec")
        self.execs.append({"command": list(command), "timeout_seconds": timeout_seconds})
        joined = " ".join(command)
        if "pip" in joined:
            if type(self).fail_install:
                return FakeRef(FakeProcessResult(returncode=1, stderr="no network"))
            return FakeRef(FakeProcessResult(returncode=0))
        if "env_server.py" in joined:
            if type(self).fail_launch:
                raise RuntimeError("exec transport blew up")
            return FakeRef(FakeProcessResult(returncode=0, stdout="4242"))
        if "tail" in joined:
            return FakeRef(FakeProcessResult(returncode=0, stdout="<server log>"))
        return FakeRef(FakeProcessResult(returncode=0))

    def write_file(self, path: str, contents: bytes) -> FakeRef:
        self.calls.append("write_file")
        if type(self).fail_write:
            return FakeRef(raises=RuntimeError("disk full"))
        self.written[path] = contents
        return FakeRef(None)

    def stop(self, **kw: Any) -> FakeRef:
        self.calls.append("stop")
        self.stopped = True
        return FakeRef(None)


class FakeNetworkOptions:
    def __init__(self, *, deny_egress=None, deny_ingress=None, egress=None, **kwargs):
        self.deny_egress = deny_egress
        self.deny_ingress = deny_ingress
        self.egress = egress
        self.kwargs = kwargs


class FakeResourceOptions:
    def __init__(self, *, requests=None, limits=None, gpu=None):
        self.requests = requests
        self.limits = limits
        self.gpu = gpu


class FakeService:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeEndpoint:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeAuthStrategy:
    WANDB = "wandb"
    COREWEAVE_API_KEY = "coreweave_api_key"


class FakeSession:
    def __init__(self, *, defaults, report_to):
        self.defaults = defaults
        self.report_to = report_to
        self.reports = []
        self.sandboxes = []

    def sandbox(self, *, command, args, **kwargs):
        assert "max_lifetime_seconds" not in kwargs
        sandbox = FakeSandbox(**self.defaults, **kwargs)
        sandbox.command = [command, *args]
        sandbox.calls.append("create")
        self.sandboxes.append(sandbox)
        return sandbox

    def log_metrics(self, *, reset=True):
        self.reports.append(reset)

    async def close(self):
        for sandbox in self.sandboxes:
            sandbox.stop().result()


@pytest.fixture
def fake_sdk(monkeypatch):
    """Install the fake `cwsandbox` module and reset failure switches."""
    FakeSandbox.instances = []
    FakeSandbox.listed = []
    FakeSandbox.list_tags = []
    FakeSandbox.list_auth = []
    for flag in (
        "fail_list",
        "fail_wait",
        "fail_service_address",
        "fail_install",
        "fail_write",
        "fail_launch",
        "fail_run",
    ):
        setattr(FakeSandbox, flag, False)

    module = types.ModuleType("cwsandbox")
    module.Sandbox = FakeSandbox
    module.NetworkOptions = FakeNetworkOptions
    module.ResourceOptions = FakeResourceOptions
    module.Service = FakeService
    module.Endpoint = FakeEndpoint
    module.AuthStrategy = FakeAuthStrategy
    module.Session = FakeSession
    monkeypatch.setitem(sys.modules, "cwsandbox", module)
    yield FakeSandbox
    FakeSandbox.instances = []
    FakeSandbox.listed = []
    FakeSandbox.list_tags = []
    FakeSandbox.list_auth = []


@pytest.fixture
def ready_http(monkeypatch):
    """Make the readiness probe succeed immediately."""

    class Resp:
        status = 200

        def read(self):
            return b'{"status":"ok"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Resp())


def _backend(**kw) -> CwSandboxBackend:
    defaults = dict(
        domain="retail",
        task_split="train",
        token="test-token",
        readiness_timeout_s=1.0,
        readiness_poll_interval_s=0.01,
    )
    defaults.update(kw)
    return CwSandboxBackend(**defaults)


# -- happy path ------------------------------------------------------------


def test_provision_call_sequence_and_handle(fake_sdk, ready_http) -> None:
    handle = _backend()._provision_blocking()
    sandbox = fake_sdk.instances[0]

    # run -> wait for RUNNING -> pip install -> upload server -> launch detached.
    # Readiness is probed over HTTP, not via exec, so there is no further exec.
    assert sandbox.calls == ["create", "wait", "exec", "write_file", "exec"], sandbox.calls
    pip = [e for e in sandbox.execs if "pip" in " ".join(e["command"])]
    assert len(pip) == 1
    assert handle.base_url == "https://10.0.0.7"
    assert handle.token == "test-token"
    assert handle.sandbox_id == sandbox.sandbox_id
    assert not sandbox.stopped
    assert "/srv/env_server.py" in sandbox.written


def test_baked_image_skips_pip_but_still_uploads_and_starts_server(fake_sdk, ready_http) -> None:
    """A custom image with tau-bench pre-installed must not run pip install."""
    handle = _backend(install_tau_bench=False, container_image="registry.example/taubench:baked")._provision_blocking()
    sandbox = fake_sdk.instances[0]

    assert sandbox.calls == ["create", "wait", "write_file", "exec"], sandbox.calls
    pip = [e for e in sandbox.execs if "pip" in " ".join(e["command"])]
    assert pip == []
    assert "/srv/env_server.py" in sandbox.written
    launch = [e for e in sandbox.execs if "env_server.py" in " ".join(e["command"])]
    assert len(launch) == 1
    assert handle.base_url == "https://10.0.0.7"


@pytest.mark.parametrize("auth,expected", [("wandb", "wandb"), ("coreweave", "coreweave_api_key")])
def test_serverless_creation_and_cleanup_share_auth(fake_sdk, ready_http, monkeypatch, auth, expected):
    monkeypatch.setenv("CWSANDBOX_SERVERLESS_AUTH", auth)
    monkeypatch.setenv("CWSANDBOX_PLACEMENT_MODE", "serverless")
    backend = _backend(placement_mode="serverless")
    backend._provision_blocking()
    cleanup_tag("test-run", settle_seconds=0)
    assert fake_sdk.instances[0].kwargs["auth"] == expected
    assert fake_sdk.list_auth == [expected]


def test_backend_reports_through_sdk_session(fake_sdk, ready_http):
    backend = _backend(max_lifetime_seconds=9876)
    backend._provision_blocking()
    backend._provision_blocking()
    assert len(backend._session.sandboxes) == 2
    assert backend._session.defaults == {"max_lifetime_seconds": 9876}
    assert backend._session.report_to == ["wandb"]
    assert backend._session.reports == [False, False]


def test_default_tau_bench_install_uses_locked_commit(fake_sdk, ready_http) -> None:
    _backend()._provision_blocking()
    pip = next(
        entry
        for entry in fake_sdk.instances[0].execs
        if "pip" in " ".join(entry["command"])
    )
    assert pip["command"][-1] == (
        "git+https://github.com/sierra-research/tau-bench.git"
        "@59a200c6d575d595120f1cb70fea53cef0632f6b"
    )


def test_network_options_declare_a_public_https_service(fake_sdk, ready_http) -> None:
    """cwsandbox 1.x exposes ports via services=, not NetworkOptions.exposed_ports."""
    handle = _backend(env_port=9001)._provision_blocking()
    sandbox = fake_sdk.instances[0]
    services = sandbox.kwargs["services"]
    assert len(services) == 1
    assert services[0].port == 9001
    assert services[0].visibility == "public"
    assert sandbox.kwargs["placement_mode"] == "cks"
    assert sandbox.kwargs["placement_spillover"] == "cks_then_serverless"
    assert handle.base_url == "https://10.0.0.7"


def test_serverless_placement_omits_cks_spillover(fake_sdk, ready_http) -> None:
    """Regression: passing placement_spillover=cks_then_serverless with
    placement_mode=serverless makes the SDK reject every create."""
    _backend(placement_mode="serverless")._provision_blocking()
    sandbox = fake_sdk.instances[0]
    assert sandbox.kwargs["placement_mode"] == "serverless"
    assert "placement_spillover" not in sandbox.kwargs
    assert sandbox.kwargs["auth"] == FakeAuthStrategy.WANDB


def test_cks_placement_keeps_default_coreweave_auth(fake_sdk, ready_http) -> None:
    _backend(placement_mode="cks")._provision_blocking()

    run_kwargs = fake_sdk.instances[0].kwargs
    assert "auth" not in run_kwargs
    assert "profile_ids" not in run_kwargs
    assert "profile_names" not in run_kwargs


def test_server_is_launched_detached_with_a_bounded_exec(fake_sdk, ready_http) -> None:
    """An attached exec would inherit the SDK's 300 s request timeout and die.

    The gRPC stream that owns an attached process is torn down at the deadline,
    so a 12-hour sandbox would lose its env server ~5 minutes in.
    """
    _backend()._provision_blocking()
    launch = [e for e in fake_sdk.instances[0].execs if "env_server.py" in " ".join(e["command"])]
    assert len(launch) == 1
    command = " ".join(launch[0]["command"])
    assert command.startswith("sh -c")
    assert "nohup" in command and command.rstrip().endswith("& echo $!")
    assert launch[0]["timeout_seconds"] is not None, "launch exec must be bounded"


def test_token_is_passed_by_env_not_argv(fake_sdk, ready_http) -> None:
    """argv is world-readable via /proc/*/cmdline inside the sandbox."""
    _backend(token="super-secret")._provision_blocking()
    sandbox = fake_sdk.instances[0]
    assert sandbox.kwargs["environment_variables"]["TAUBENCH_ENV_TOKEN"] == "super-secret"
    for entry in sandbox.execs:
        assert "super-secret" not in " ".join(entry["command"])
        assert "--token" not in " ".join(entry["command"])


def test_domain_and_split_are_shell_quoted(fake_sdk, ready_http) -> None:
    _backend(domain="retail", task_split="tr ain; rm -rf /")._provision_blocking()
    launch = [e for e in fake_sdk.instances[0].execs if "env_server.py" in " ".join(e["command"])][0]
    command = " ".join(launch["command"])
    assert "'tr ain; rm -rf /'" in command


# -- failure paths must never leak a running sandbox -----------------------


@pytest.mark.parametrize(
    "flag,match",
    [
        ("fail_wait", "did not reach RUNNING"),
        ("fail_service_address", "service_urls"),
        ("fail_install", "tau-bench install failed"),
        ("fail_write", "disk full"),
        ("fail_launch", "exec transport blew up"),
    ],
)
def test_every_provision_failure_stops_the_sandbox(fake_sdk, ready_http, flag, match) -> None:
    """A leaked sandbox bills for its whole max_lifetime_seconds (12 h default)."""
    setattr(fake_sdk, flag, True)
    with pytest.raises(Exception, match=match):
        _backend()._provision_blocking()

    assert len(fake_sdk.instances) == 1
    assert fake_sdk.instances[0].stopped is True, f"{flag} leaked a running sandbox"


def test_readiness_timeout_stops_the_sandbox_and_reports_the_log(fake_sdk, monkeypatch) -> None:
    """A server that never binds must fail loudly, with diagnosis, and clean up."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError("refused")),
    )

    with pytest.raises(TimeoutError) as exc:
        _backend()._provision_blocking()

    assert "did not become ready" in str(exc.value)
    assert "<server log>" in str(exc.value), "failure must include the server log tail"
    assert fake_sdk.instances[0].stopped is True


def test_readiness_401_fails_fast_without_retrying(fake_sdk, monkeypatch) -> None:
    """A token mismatch cannot be fixed by waiting; retrying just burns the timeout."""
    calls = {"n": 0}

    def raise_401(*a, **k):
        calls["n"] += 1
        raise urllib.error.HTTPError("http://x/health", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_401)

    with pytest.raises(RuntimeError, match="rejected our bearer token"):
        _backend(readiness_timeout_s=5.0)._provision_blocking()

    assert calls["n"] == 1, "401 must not be retried"
    assert fake_sdk.instances[0].stopped is True


@pytest.mark.asyncio
async def test_async_provision_warms_sdk_on_main_thread_before_worker_thread(
    fake_sdk, ready_http, monkeypatch
) -> None:
    """A normal async caller imports the SDK before blocking work enters to_thread."""
    sdk_threads: list[threading.Thread] = []
    blocking_threads: list[threading.Thread] = []
    real_sdk = CwSandboxBackend._sdk
    real_blocking = CwSandboxBackend._provision_blocking

    def tracking_sdk(self):
        sdk_threads.append(threading.current_thread())
        return real_sdk(self)

    def tracking_blocking(self):
        blocking_threads.append(threading.current_thread())
        return real_blocking(self)

    monkeypatch.setattr(CwSandboxBackend, "_sdk", tracking_sdk)
    monkeypatch.setattr(CwSandboxBackend, "_provision_blocking", tracking_blocking)

    backend = _backend()
    await backend.provision()

    assert sdk_threads[0] is threading.main_thread()
    assert blocking_threads[0] is not threading.main_thread()


def test_sdk_import_skips_signal_registration_on_ray_actor_thread(monkeypatch) -> None:
    """Ray async actors construct tools off-main, where signal.signal is forbidden."""
    sdk = types.SimpleNamespace(
        Sandbox=FakeSandbox,
        NetworkOptions=FakeNetworkOptions,
        ResourceOptions=FakeResourceOptions,
        AuthStrategy=FakeAuthStrategy,
    )
    original_signal = signal.signal
    imported_on: list[threading.Thread] = []
    errors: list[BaseException] = []

    def fake_import_module(name: str):
        assert name == "cwsandbox"
        imported_on.append(threading.current_thread())
        signal.signal(signal.SIGINT, lambda *_args: None)
        return sdk

    def preload() -> None:
        try:
            _backend().preload_sdk()
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(
        "verl_taubench.sandbox.pool.importlib.import_module",
        fake_import_module,
    )
    worker = threading.Thread(target=preload)
    worker.start()
    worker.join()

    assert errors == []
    assert imported_on == [worker]
    assert signal.signal is original_signal


@pytest.mark.asyncio
async def test_destroy_stops_the_sandbox(fake_sdk, ready_http) -> None:
    backend = _backend()
    handle = await backend.provision()
    await backend.destroy(handle)
    assert fake_sdk.instances[0].stopped is True


def test_cleanup_reaps_only_the_exact_requested_tag(fake_sdk) -> None:
    first = FakeSandbox()
    second = FakeSandbox()
    fake_sdk.listed = [first, second]

    assert cleanup_tag("verl-taubench-env-run-123", settle_seconds=0) == 2
    assert all(tags == ["verl-taubench-env-run-123"] for tags in fake_sdk.list_tags)
    assert first.stopped is True
    assert second.stopped is True


def test_cleanup_starts_all_stops_before_waiting(fake_sdk) -> None:
    """Regression: sequential waits made large pool teardown appear hung."""
    events: list[str] = []

    class StopRef:
        def __init__(self, owner: "ListedSandbox"):
            self.owner = owner

        def result(self) -> None:
            assert events == ["start-1", "start-2", "start-3"]
            self.owner.stopped = True

    class ListedSandbox:
        def __init__(self, index: int):
            self.index = index
            self.stopped = False

        def stop(self) -> StopRef:
            events.append(f"start-{self.index}")
            return StopRef(self)

    fake_sdk.listed = [ListedSandbox(1), ListedSandbox(2), ListedSandbox(3)]

    assert _backend().reap_orphans() == 3
    assert all(sandbox.stopped for sandbox in fake_sdk.listed)


def test_cleanup_uses_wandb_auth_for_serverless_placement(fake_sdk, monkeypatch) -> None:
    monkeypatch.setenv("CWSANDBOX_PLACEMENT_MODE", "serverless")

    cleanup_tag("verl-taubench-env-run-123", settle_seconds=0)

    assert fake_sdk.list_auth == [FakeAuthStrategy.WANDB]


def test_cleanup_keeps_default_coreweave_auth_for_cks_placement(fake_sdk, monkeypatch) -> None:
    monkeypatch.setenv("CWSANDBOX_PLACEMENT_MODE", "cks")

    cleanup_tag("verl-taubench-env-run-123", settle_seconds=0)

    assert fake_sdk.list_auth == [None]


def test_cleanup_surfaces_list_auth_failures(fake_sdk, monkeypatch) -> None:
    monkeypatch.setenv("CWSANDBOX_PLACEMENT_MODE", "serverless")
    fake_sdk.fail_list = True

    with pytest.raises(RuntimeError, match="Sandbox.list"):
        cleanup_tag("verl-taubench-env-run-123", settle_seconds=0)


def test_cleanup_sweeps_until_late_arriving_creates_are_reaped(fake_sdk) -> None:
    # A provision retry still in flight when the trainer exits lands AFTER the
    # first reap pass listed; the settle loop must catch it.
    first = FakeSandbox()
    late = FakeSandbox()
    fake_sdk.listed = [first]

    original_list = fake_sdk.list.__func__

    def listing_with_late_arrival(cls, *, tags, auth=None):
        result = original_list(cls, tags=tags, auth=auth)
        if len(cls.list_tags) == 1:  # after the first pass, the orphan lands
            cls.listed = [first, late]
        return result

    fake_sdk.list = classmethod(listing_with_late_arrival)
    try:
        assert cleanup_tag("verl-taubench-env-run-123", settle_seconds=0) == 2
    finally:
        fake_sdk.list = classmethod(original_list)
    assert first.stopped is True
    assert late.stopped is True
    # Terminates once a follow-up pass finds nothing (not the pass budget).
    assert len(fake_sdk.list_tags) == 3


def test_cleanup_rejects_empty_tag_instead_of_broad_reaping(fake_sdk) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cleanup_tag("")
    assert fake_sdk.list_tags == []


def test_runner_ids_are_passed_to_sandbox_run(fake_sdk, ready_http) -> None:
    _backend(runner_ids=["cluster-runner-1"])._provision_blocking()
    assert fake_sdk.instances[0].kwargs["runner_ids"] == ["cluster-runner-1"]
    assert fake_sdk.instances[0].kwargs["placement_mode"] == "cks"



def test_env_sandboxes_are_created_with_the_stable_recipe_tag(fake_sdk, ready_http) -> None:
    """A leaked env sandbox must be identifiable without the run id."""
    from verl_taubench.sandbox.tags import RECIPE_TAG

    _backend(tags=["verl-taubench-env-run-77"])._provision_blocking()
    tags = fake_sdk.instances[0].kwargs["tags"]
    assert RECIPE_TAG in tags, "cleanup --all relies on this tag"
    assert "verl-taubench-env-run-77" in tags, "per-run reaping relies on this tag"


def test_reaping_lists_exactly_the_requested_tag(fake_sdk) -> None:
    """Reap queries must not be widened by the creation-time recipe tag."""
    _backend(tags=["verl-taubench-env-run-77"]).reap_orphans()
    assert fake_sdk.list_tags == [["verl-taubench-env-run-77"]]
