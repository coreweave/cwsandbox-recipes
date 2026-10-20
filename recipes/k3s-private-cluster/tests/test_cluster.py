import argparse
from types import SimpleNamespace

import pytest
import yaml

import cluster


@pytest.mark.parametrize("value", ["2", "0.5", "500m"])
def test_cpu_accepts_kubernetes_quantities(value):
    assert cluster.cpu_quantity(value) == value


@pytest.mark.parametrize("value", ["0", "-1", "1; echo bad", "1Gi"])
def test_cpu_rejects_invalid_resources(value):
    with pytest.raises(argparse.ArgumentTypeError):
        cluster.cpu_quantity(value)


@pytest.mark.parametrize("value", ["0Gi", "-4Gi", "8Gi; echo bad", "eight"])
def test_memory_rejects_invalid_resources(value):
    with pytest.raises(argparse.ArgumentTypeError):
        cluster.memory_quantity(value)


def test_node_command_uses_secret_files_and_guest_network():
    command = cluster.node_command({"name": "k3s-worker-2", "ip": "100.64.0.3"}, "100.64.0.1")
    assert "--token-file /demo/node-token" in command
    assert "--vpn-auth-file /demo/vpn-auth" in command
    assert "--server https://100.64.0.1:6443" in command
    assert "--network host" in command
    assert "joinKey=" not in command


def test_node_command_can_pin_a_compatible_kubernetes_version():
    image = "rancher/k3s:v1.35.1-k3s1"
    command = cluster.node_command({"name": "k3s-server", "ip": "100.64.0.1"}, image=image)
    assert image in command
    assert cluster.K3S_IMAGE not in command


@pytest.mark.parametrize("value", ["latest", "v1.35.1-k3s1; echo bad", "other/image:v1"])
def test_k3s_version_rejects_unpinned_or_arbitrary_images(value):
    with pytest.raises(argparse.ArgumentTypeError):
        cluster.k3s_version(value)


def test_readiness_requires_exact_nodes_and_ready_conditions():
    node = {
        "metadata": {"name": "one"},
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }
    assert cluster.ready_nodes({"items": [node]}, {"one"})
    assert not cluster.ready_nodes({"items": [node]}, {"one", "two"})
    node["status"]["conditions"][0]["status"] = "False"
    assert not cluster.ready_nodes({"items": [node]}, {"one"})


def test_failed_drain_preserves_worker_and_sandbox(monkeypatch):
    state = {"workers": {"worker-1": {"sandbox_id": "owned"}}, "removed": []}
    stopped = []
    monkeypatch.setattr(cluster, "server_for", lambda _: object())
    monkeypatch.setattr(cluster, "stop_node", lambda n: stopped.append(n))

    def failed_drain(*args):
        raise RuntimeError("disruption budget blocks eviction")

    monkeypatch.setattr(cluster, "kube", failed_drain)
    with pytest.raises(RuntimeError, match="disruption budget"):
        cluster.remove_worker(state, "worker-1")
    assert "worker-1" in state["workers"]
    assert not stopped


def test_down_attempts_every_owned_sandbox_after_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster, "STATE", tmp_path / "state.json")
    cluster.save({"workers": {"worker-1": {"name": "worker-1"}}, "server": {"name": "server"}})
    attempted = []

    def stop(node):
        attempted.append(node["name"])
        if node["name"] == "worker-1":
            raise RuntimeError("transient")

    monkeypatch.setattr(cluster, "stop_node", stop)
    with pytest.raises(RuntimeError, match="Cleanup incomplete"):
        cluster.down(SimpleNamespace())
    assert attempted == ["worker-1", "server"]
    assert cluster.STATE.exists()


def test_workload_spreads_replicas_and_labels_response_with_node():
    deployment, service = list(yaml.safe_load_all((cluster.ROOT / "workloads.yaml").read_text()))
    pod = deployment["spec"]["template"]
    terms = pod["spec"]["affinity"]["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]
    assert terms[0]["topologyKey"] == "kubernetes.io/hostname"
    assert service["spec"]["selector"] == pod["metadata"]["labels"]
    assert (
        pod["spec"]["containers"][0]["env"][0]["valueFrom"]["fieldRef"]["fieldPath"]
        == "spec.nodeName"
    )


def test_scale_uses_requested_size_only_for_new_workers(monkeypatch):
    state = {"workers": {"old": {"ready": True, "cpu": "2", "memory": "4Gi"}}}
    monkeypatch.setattr(cluster, "load", lambda: state)
    monkeypatch.setattr(cluster, "status", lambda _: None)

    def add(current, cpu, memory):
        current["workers"]["new"] = {"ready": True, "cpu": cpu, "memory": memory}

    monkeypatch.setattr(cluster, "add_worker", add)
    cluster.scale(SimpleNamespace(workers=2, cpu="4", memory="8Gi", test=False))
    assert state["workers"]["old"]["cpu"] == "2"
    assert state["workers"]["old"]["memory"] == "4Gi"
    assert state["workers"]["new"]["cpu"] == "4"
    assert state["workers"]["new"]["memory"] == "8Gi"


def test_scale_down_sample_requires_explicit_test_flag(monkeypatch):
    state = {"workers": {"old": {"ready": True}, "new": {"ready": True}}}
    calls = []
    monkeypatch.setattr(cluster, "load", lambda: state)
    monkeypatch.setattr(cluster, "server_for", lambda _: None)
    monkeypatch.setattr(cluster, "kube", lambda _, *args: calls.append(args))
    monkeypatch.setattr(cluster, "status", lambda _: None)
    monkeypatch.setattr(cluster, "test", lambda _: calls.append(("test",)))

    def remove(current, name):
        calls.append(("remove", name))
        del current["workers"][name]

    monkeypatch.setattr(cluster, "remove_worker", remove)
    cluster.scale(SimpleNamespace(workers=1, test=True))
    assert calls == [
        ("scale", "deployment/web", "--replicas=2"),
        ("rollout", "status", "deployment/web", "--timeout=180s"),
        ("remove", "new"),
        ("test",),
    ]
    calls.clear()
    cluster.scale(SimpleNamespace(workers=0, test=False))
    assert calls == [("remove", "old")]
