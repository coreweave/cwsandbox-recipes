"""Async GRPO: the generation object the collector actor receives, the config and its selection."""

import json
import pickle
import sys
import threading
import types
from pathlib import Path

import pytest
import yaml

from async_grpo_tool_use import dapo_data, run_trainer

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def generation_cls(monkeypatch):
    """HotLoadGeneration with a stand-in for NeMo-RL's GenerationInterface."""
    interfaces = types.ModuleType("nemo_rl.models.generation.interfaces")
    interfaces.GenerationInterface = type("GenerationInterface", (), {})
    for name in ("nemo_rl", "nemo_rl.models", "nemo_rl.models.generation"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nemo_rl.models.generation.interfaces", interfaces)
    monkeypatch.delitem(sys.modules, "async_grpo_tool_use.nemo_rl_ext.generation", raising=False)
    from async_grpo_tool_use.nemo_rl_ext.generation import HotLoadGeneration

    return HotLoadGeneration


def _generation(cls):
    cfg = {
        "base_url": "https://gw.example/v1",
        "model": "m",
        "deployment_id": "dep",
        "run_id": "run",
        "max_model_len": 3072,
    }
    return cls({"backend": "coreweave_hotload", "max_new_tokens": 2048, "hotload_cfg": cfg})


def test_collector_copy_leaves_out_the_weight_synchronizer(generation_cls):
    generation = _generation(generation_cls)
    # The real synchronizer holds API and S3 clients, which do not pickle either.
    generation.weight_synchronizer = threading.Lock()

    copy = pickle.loads(pickle.dumps(generation))

    assert copy.weight_synchronizer is None
    assert copy.cfg["max_total_sequence_length"] == 3072
    assert copy.base_url == "https://gw.example/v1"
    assert generation.weight_synchronizer is not None  # the trainer's object keeps it


def test_kv_cache_invalidation_reports_success(generation_cls):
    assert _generation(generation_cls).invalidate_kv_cache() is True


def _load(name):
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def test_async_config_passes_nemo_rl_async_checks():
    cfg = _load("grpo_async_qwen.yaml")
    grpo = cfg["grpo"]
    # run_grpo_nemo_gym.py raises NotImplementedError for these in async mode.
    assert not grpo["use_dynamic_sampling"]
    assert not grpo["reward_scaling"]["enabled"]
    assert not grpo["reward_shaping"]["enabled"]
    # async_grpo_train asserts importance sampling correction.
    assert cfg["loss_fn"]["use_importance_sampling_correction"]
    assert grpo["async_grpo"] == {
        "enabled": True,
        "max_trajectory_age_steps": 1,
        "in_flight_weight_updates": True,
        "recompute_kv_cache_after_weight_updates": False,
    }
    # One train step consumes exactly the prompt groups the replay buffer hands over.
    assert cfg["policy"]["train_global_batch_size"] == grpo["num_prompts_per_step"] * grpo["num_generations_per_prompt"]


def test_trainer_runs_the_async_config():
    cmd = run_trainer.grpo_command([], {})
    assert cmd[cmd.index("--config") + 1] == "/workspace/config/grpo_async_qwen.yaml"


def test_rollouts_reach_both_tool_use_agents():
    agents = _load("grpo_async_qwen.yaml")["env"]["nemo_gym"]["remote_agent_urls"]
    assert set(agents) == {dapo_data.TRAIN_AGENT, dapo_data.EVAL_AGENT}


@pytest.fixture
def weight_sync(monkeypatch):
    """The weight_sync module with a stand-in for NeMo-RL's WeightSynchronizer."""
    interfaces = types.ModuleType("nemo_rl.weight_sync.interfaces")
    interfaces.WeightSynchronizer = type("WeightSynchronizer", (), {})
    for name in ("nemo_rl", "nemo_rl.weight_sync"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nemo_rl.weight_sync.interfaces", interfaces)
    monkeypatch.delitem(sys.modules, "async_grpo_tool_use.nemo_rl_ext.weight_sync", raising=False)
    from async_grpo_tool_use.nemo_rl_ext import weight_sync

    monkeypatch.setattr(weight_sync.time, "sleep", lambda s: None)
    return weight_sync


def test_hot_load_waits_for_a_replica_change_to_roll_out(weight_sync):
    from cwtrain.errors import FailedPreconditionError

    calls = []

    def publish():
        calls.append(1)
        if len(calls) < 3:
            raise FailedPreconditionError(
                "failed_precondition", "deployment is not ready; wait for the rollout to converge before hot-loading"
            )

    weight_sync.publish_when_ready(publish, 12)
    assert len(calls) == 3


def test_hot_load_does_not_retry_other_failed_preconditions(weight_sync):
    from cwtrain.errors import FailedPreconditionError

    def publish():
        raise FailedPreconditionError("failed_precondition", "hot-load path is outside the bucket prefix")

    with pytest.raises(FailedPreconditionError):
        weight_sync.publish_when_ready(publish, 1)


class FakeLoader:
    """cwtrain's HotLoader: the upload reads the export dir, and the swap can be held or fail."""

    def __init__(self, log):
        self.log = log
        self.s3 = "s3"
        self.deployment = types.SimpleNamespace(hot_load=types.SimpleNamespace(bucket="b", path_prefix="/p/"))
        self.release = threading.Event()
        self.release.set()
        self.fail: Exception | None = None
        self.not_ready = 0  # trigger calls to reject while a replica change rolls out

    def upload(self, path, identity, workers):
        assert Path(path).is_dir(), "the export dir must stay until the upload is done"
        self.log.append(("upload", identity))
        return identity

    def trigger(self, identity, info, prompt_cache_policy):
        from cwtrain.errors import FailedPreconditionError

        if self.not_ready:
            self.not_ready -= 1
            raise FailedPreconditionError("failed_precondition", "deployment is not ready")
        return types.SimpleNamespace(id=identity)

    def wait(self, hot_load_id):
        assert self.release.wait(5)
        if self.fail:
            raise self.fail
        self.log.append(("served", hot_load_id))


class FakePolicy:
    def __init__(self, log):
        self.log = log

    def run_all_workers_single_data(self, method, path, dtypes, sizes):
        assert method == "export_hf_checkpoint" and sizes == {"a": 2}
        self.log.append(("export", Path(path).name))
        Path(path).mkdir(parents=True)
        return [{"rank": 0, "tensors": 2, "bytes": 2**29}, {"rank": 1, "tensors": 1, "bytes": 2**29}]


@pytest.fixture
def make_sync(weight_sync, monkeypatch, tmp_path):
    """A HotLoadWeightSynchronizer over fakes; ``log`` records export, upload, swap and delete order."""
    import cwtrain

    log: list = []
    loader = FakeLoader(log)
    monkeypatch.setenv("COREWEAVE_API_TOKEN", "t")
    monkeypatch.setattr(weight_sync, "checkpoint_dtypes", lambda base: {"a": "BF16"})
    monkeypatch.setattr(weight_sync, "checkpoint_sizes", lambda base: {"a": 2})
    monkeypatch.setattr(weight_sync, "add_model_files", lambda base, path: None)
    monkeypatch.setattr(weight_sync.caios, "client", lambda: "s3")
    monkeypatch.setattr(weight_sync.caios, "delete_prefix", lambda s3, bucket, prefix: log.append(("delete", prefix)))
    monkeypatch.setattr(cwtrain, "InferenceClient", lambda token: None)
    monkeypatch.setattr(cwtrain, "HotLoader", lambda client, deployment_id, s3, poll_interval: loader)

    def make(behind, async_grpo=True):
        generation = types.SimpleNamespace(
            hotload_cfg={"deployment_id": "dep", "run_id": "run", "max_versions_behind": behind}
        )
        sync = weight_sync.HotLoadWeightSynchronizer(
            FakePolicy(log), generation, hf_checkpoint="/base", export_dir=str(tmp_path), async_grpo=async_grpo
        )
        return sync, loader, log

    return make


def _in_thread(fn):
    thread = threading.Thread(target=fn)
    thread.start()
    thread.join(0.2)
    return thread


def test_trainer_never_waits_and_only_the_newest_export_is_loaded(make_sync, tmp_path):
    sync, loader, log = make_sync(behind=2)
    loader.release.clear()
    for _ in range(2):  # v1 swaps slowly; v2 is exported meanwhile
        assert not _in_thread(sync.sync_weights).is_alive(), "within 2 versions the trainer must not wait"
    third = _in_thread(sync.sync_weights)
    assert third.is_alive(), "v3 would leave the deployment 3 versions behind"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v000001", "v000003"]  # v3 replaced v2
    loader.release.set()
    third.join(5)
    sync.shutdown()

    assert [e for e in log if e[0] != "export"] == [
        ("upload", "run-v000001"), ("served", "run-v000001"),
        ("upload", "run-v000003"), ("served", "run-v000003"),  # v2 was never uploaded
        ("delete", "p/run-v000001"),  # the previous snapshot goes only once the next one serves
    ]  # fmt: skip
    assert not any(tmp_path.iterdir()), "each export dir is removed after its upload or when dropped"


def test_unbounded_lag_never_waits_and_loads_only_the_newest_export(make_sync, tmp_path):
    sync, loader, log = make_sync(behind=float("inf"))
    loader.release.clear()
    for _ in range(5):  # v1 swaps slowly; v2..v5 are exported meanwhile
        assert not _in_thread(sync.sync_weights).is_alive(), "the trainer must never wait"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v000001", "v000005"]  # v2..v4 dropped
    loader.release.set()
    sync.shutdown()
    assert [e for e in log if e[0] == "upload"] == [("upload", "run-v000001"), ("upload", "run-v000005")]
    assert not any(tmp_path.iterdir())


def test_one_version_behind_overlaps_the_export_with_the_previous_upload(make_sync):
    sync, loader, log = make_sync(behind=1)
    loader.release.clear()
    sync.sync_weights()
    second = _in_thread(sync.sync_weights)
    assert ("export", "v000002") in log, "the export runs while the previous hot-load swaps"
    assert second.is_alive(), "and then the trainer waits for it"
    loader.release.set()
    second.join(5)
    sync.shutdown()
    assert ("served", "run-v000002") in log


def test_blocking_sync_returns_once_the_deployment_serves_the_weights(make_sync):
    sync, loader, log = make_sync(behind=0, async_grpo=False)
    loader.not_ready = 2
    sync.sync_weights()
    assert log == [("export", "v000001"), ("upload", "run-v000001"), ("served", "run-v000001")]
    sync.shutdown()


def test_hot_load_error_raises_in_the_trainer_once_then_stops_the_run(make_sync):
    sync, loader, log = make_sync(behind=2)
    loader.fail = RuntimeError("hot-load failed")
    sync.sync_weights()
    sync._publisher.join(5)
    with pytest.raises(RuntimeError, match="hot-load failed"):
        sync.sync_weights()
    with pytest.raises(RuntimeError, match="stopped after an earlier error"):
        sync.wait()
    sync.shutdown()  # the error was raised already; shutdown only cleans up
    assert ("export", "v000002") not in log


def test_shutdown_waits_for_the_last_hot_load_and_raises_its_error(make_sync):
    sync, loader, log = make_sync(behind=2)
    loader.release.clear()
    sync.sync_weights()
    threading.Timer(0.1, loader.release.set).start()
    sync.shutdown()
    assert log[-1] == ("served", "run-v000001")

    sync, loader, log = make_sync(behind=2)
    loader.fail = RuntimeError("hot-load failed")
    sync.sync_weights()
    with pytest.raises(RuntimeError, match="hot-load failed"):
        sync.shutdown()


def test_hot_load_prints_dashboard_events(make_sync, capsys):
    for behind in (0, 2):
        sync, _, _ = make_sync(behind=behind)
        sync.sync_weights()
        sync.shutdown()
        out = capsys.readouterr().out
        events = [json.loads(line.split("@@metric ", 1)[1]) for line in out.splitlines() if "@@metric " in line]
        assert [(e["k"], e["version"], e["state"]) for e in events] == [("hotload", 1, "start"), ("hotload", 1, "done")]
        assert events[0]["ts"] <= events[1]["ts"]
        [done] = [line for line in out.splitlines() if "upload and load" in line]
        assert done.endswith("in the background") == bool(behind)
        assert "3 tensors, 1.0 GiB" in done  # summed over the ranks' shards

    sync, loader, _ = make_sync(behind=2)
    loader.release.clear()
    sync.sync_weights()
    sync.sync_weights()
    third = _in_thread(sync.sync_weights)
    loader.release.set()
    third.join(5)
    sync.shutdown()
    out = capsys.readouterr().out
    assert '"version": 2, "state": "skipped"' in out and "hot-load v2: skipped, v3 is newer" in out


def test_lagging_hot_loads_are_refused_for_synchronous_grpo(make_sync):
    with pytest.raises(ValueError, match="async GRPO"):
        make_sync(behind=1, async_grpo=False)


def test_validation_waits_for_the_newest_weights(generation_cls):
    generation = _generation(generation_cls)
    waits = []
    generation.weight_synchronizer = types.SimpleNamespace(wait=lambda: waits.append(1))
    assert generation.prepare_for_generation() is True
    assert waits == [1]
    # The collector's copy has no synchronizer and does not wait.
    assert pickle.loads(pickle.dumps(generation)).prepare_for_generation() is True


def test_hot_loads_may_lag_the_trainer():
    assert _load("grpo_async_qwen.yaml")["policy"]["generation"]["hotload_cfg"]["max_versions_behind"] == float("inf")


def test_uploads_use_larger_parts_and_everything_else_passes_through(weight_sync):
    calls = []
    client = types.SimpleNamespace(
        upload_file=lambda *args, **kwargs: calls.append((args, kwargs)), head_object=lambda **kw: "head"
    )
    s3 = weight_sync.TunedS3(client)
    s3.upload_file("f", "b", "k", ExtraArgs={"Metadata": {"cwtrain-sha256": "x"}})
    [(args, kwargs)] = calls
    assert args == ("f", "b", "k") and kwargs["ExtraArgs"] == {"Metadata": {"cwtrain-sha256": "x"}}
    assert (kwargs["Config"].multipart_chunksize, kwargs["Config"].max_concurrency) == (32 * 2**20, 16)
    assert s3.head_object(Bucket="b") == "head"
