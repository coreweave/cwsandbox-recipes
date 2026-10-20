"""Cold control-plane recovery using a CoreWeave filesystem snapshot."""

import json
import time
import uuid
from datetime import datetime

BACKUP = """
set -eu
umask 077
docker stop --time 30 k3s >/dev/null
pkill -x headscale || true
pkill -x tailscaled || true
for n in $(seq 1 30); do
  if ! pidof headscale >/dev/null && ! pidof tailscaled >/dev/null; then break; fi
  sleep 1
done
! pidof headscale >/dev/null
! pidof tailscaled >/dev/null
mkdir -p /checkpoint/k3s /checkpoint/headscale /checkpoint/config /checkpoint/node
cp -a /var/lib/rancher/k3s/cluster/server /checkpoint/k3s/
cp -a /var/lib/headscale/. /checkpoint/headscale/
cp -a /etc/headscale/. /checkpoint/config/
cp -a /demo/node/. /checkpoint/node/
cp /demo/authkey /demo/vpn-auth /demo/tailscaled.state /checkpoint/
sync
"""


def start_network(server, api):
    api.execute(
        server,
        "if ! pidof headscale >/dev/null; then "
        "nohup headscale serve >/demo/headscale.log 2>&1 </dev/null & fi",
    )
    api.wait_command(server, "curl -fsS http://127.0.0.1:8080/health")
    api.execute(
        server,
        "if ! pidof tailscaled >/dev/null; then nohup tailscaled --state=/demo/tailscaled.state "
        "--socket=/var/run/tailscale/tailscaled.sock "
        ">/demo/tailscaled.log 2>&1 </dev/null & fi",
    )
    api.wait_command(server, "tailscale ip -4")


def identities(server, api):
    resources = {
        "namespace": ["namespace", "default"],
        "deployment": ["deployment", "web"],
        "service": ["service", "web"],
        "marker": ["configmap", "restore-marker"],
    }
    result = {}
    for key, args in resources.items():
        obj = json.loads(api.kube(server, "get", *args, "-o", "json"))
        result[key] = obj["metadata"]["uid"]
        if key == "marker":
            result["marker_value"] = obj["data"]["value"]
    return result


def snapshot(args, api):
    state = api.load()
    if state.get("checkpoint"):
        raise RuntimeError("This demo already has a checkpoint; restore it or run down.")
    server = api.server_for(state)
    # Deliberately create data that is absent from workloads.yaml.
    marker = uuid.uuid4().hex
    api.kube(server, "create", "configmap", "restore-marker", f"--from-literal=value={marker}")
    manifest = {
        "schema": 1,
        "source_sandbox_id": state["server"]["sandbox_id"],
        "server_ip": state["server"]["ip"],
        "identities": identities(server, api),
        "workers": {k: n["sandbox_id"] for k, n in state["workers"].items()},
    }
    checkpoint = {"manifest": manifest, "phase": "preparing", "request_id": str(uuid.uuid4())}
    state["checkpoint"] = checkpoint
    api.save(state)
    print("Pausing K3s and saving the control plane...", flush=True)
    try:
        api.execute(server, BACKUP)
        server.write_file("/checkpoint/manifest.json", json.dumps(manifest).encode()).result()
    finally:
        # The immutable staging directory is separate from the running network state.
        start_network(server, api)
    checkpoint["phase"] = "prepared"
    checkpoint["prepared_at"] = time.time()
    api.save(state)
    if getattr(args, "prepare_only", False):
        print("Checkpoint prepared. K3s is paused.", flush=True)
        return
    snapshot_id = server.snapshot(
        wait_for_ready=False, request_id=checkpoint["request_id"]
    ).result()
    checkpoint["snapshot_id"] = snapshot_id
    api.save(state)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        saved = api.Sandbox.get_snapshot(snapshot_id).result()
        status = saved.status.value.lower()
        if status == "ready":
            checkpoint["phase"] = "ready"
            api.save(state)
            print(f"Snapshot ready: {snapshot_id}", flush=True)
            print("K3s is paused. Run restore, or resume the original control plane.", flush=True)
            return
        if status == "failed":
            raise RuntimeError("Filesystem snapshot failed; resume the original or run down.")
        time.sleep(2)
    raise TimeoutError("Snapshot is still pending; its ID is retained in state.")


def resume(args, api):
    state = api.load()
    server = api.server_for(state)
    api.execute(server, "docker start k3s >/dev/null")
    api.wait_node(server, "k3s-server")
    print("Original control plane resumed.", flush=True)


def wait_for_fresh_leases(state, server, since, api):
    """Snapshot Ready conditions are historical until kubelets report again."""
    pending = {"k3s-server", *state["workers"]}
    deadline = time.monotonic() + 120
    while pending and time.monotonic() < deadline:
        leases = json.loads(
            api.kube(server, "get", "leases", "-n", "kube-node-lease", "-o", "json")
        )
        for lease in leases["items"]:
            renewed = lease.get("spec", {}).get("renewTime")
            if (
                renewed
                and datetime.fromisoformat(renewed.replace("Z", "+00:00")).timestamp() > since
            ):
                pending.discard(lease["metadata"]["name"])
        if pending:
            time.sleep(2)
    if pending:
        raise TimeoutError("Kubelets have not renewed their leases after restore.")


def restore(args, api):
    state = api.load()
    checkpoint = state.get("checkpoint", {})
    snapshot_id = getattr(args, "snapshot", None) or checkpoint.get("snapshot_id")
    if checkpoint.get("phase") not in {"prepared", "ready"} or not snapshot_id:
        raise RuntimeError("Prepare a checkpoint and create a ready snapshot before restoring.")
    saved = api.Sandbox.get_snapshot(snapshot_id).result()
    if saved.status.value.lower() != "ready":
        raise RuntimeError("The checkpoint snapshot is not ready.")
    if saved.source_sandbox_id != checkpoint["manifest"]["source_sandbox_id"]:
        raise RuntimeError("The snapshot belongs to a different source sandbox.")
    if saved.created_at.timestamp() < checkpoint.get("prepared_at", 0):
        raise RuntimeError("The snapshot predates the prepared checkpoint.")
    checkpoint["snapshot_id"] = snapshot_id
    api.save(state)
    manifest = checkpoint["manifest"]
    if manifest["workers"] != {k: n["sandbox_id"] for k, n in state["workers"].items()}:
        raise RuntimeError(
            "Worker membership changed after the snapshot; cannot restore this demo."
        )
    if int(state["expires_at"] - time.time()) < 300:
        raise RuntimeError("Less than five minutes remain; create a fresh demo.")
    original = state["server"].copy()
    # Fence the original before reusing its Kubernetes and VPN identity.
    api.stop_node(original)
    state["removed"].append(original)
    checkpoint["phase"] = "restoring"
    api.save(state)
    server, node = api.new_sandbox(
        state, "k3s-server", original["cpu"], original["memory"], snapshot_id
    )
    if node["sandbox_id"] == original["sandbox_id"]:
        raise RuntimeError("Expected a new sandbox ID after restore.")
    print(f"Restored sandbox: {node['sandbox_id']}", flush=True)
    restored_manifest = json.loads(server.read_file("/checkpoint/manifest.json").result())
    if restored_manifest != manifest:
        raise RuntimeError("Restored checkpoint manifest does not match the expected cluster.")
    api.execute(
        server,
        """
        mkdir -p /etc/headscale /var/lib/headscale /var/run/headscale
        cp -a /checkpoint/k3s/server /var/lib/rancher/k3s/cluster/
        cp -a /checkpoint/headscale/. /var/lib/headscale/
        cp -a /checkpoint/config/. /etc/headscale/
        cp -a /checkpoint/node/. /demo/node/
        cp /checkpoint/authkey /checkpoint/vpn-auth /demo/
        curl -fsSL --retry 3 -o /usr/local/bin/headscale \
          https://github.com/juanfont/headscale/releases/download/v0.29.5/headscale_0.29.5_linux_amd64
        chmod +x /usr/local/bin/headscale
        nohup headscale serve >/demo/headscale.log 2>&1 </dev/null &
    """,
    )
    api.wait_command(server, "curl -fsS http://127.0.0.1:8080/health")
    node["ip"] = api.wait_command(server, "tailscale ip -4").strip()
    if node["ip"] != manifest["server_ip"]:
        raise RuntimeError("Restored VPN address differs from the original control plane.")
    node["network_id"] = original["network_id"]
    api.save(state)
    api.execute(
        server,
        "nohup tailcat --json serve --key=new --full-address 8080 "
        ">/demo/tailcat.json 2>/demo/tailcat.log </dev/null &",
    )
    api.wait_command(server, "test -s /demo/tailcat.json")
    address = json.loads(server.read_file("/demo/tailcat.json").result())["listenAddr"]
    for worker in state["workers"].values():
        sb = api.attach(worker)
        sb.write_file("/demo/headscale.addr", address.encode()).result()
        api.execute(sb, "pkill -x tailcat || true")
        api.execute(
            sb,
            "nohup tailcat --key=new forward --bind=127.0.0.1 "
            '"$(cat /demo/headscale.addr)" 8080:8080 '
            ">/demo/tailcat.log 2>&1 </dev/null &",
        )
        api.wait_command(sb, "curl -fsS http://127.0.0.1:8080/health")
    started = time.time()
    api.execute(server, api.node_command(node, image=state.get("k3s_image", api.K3S_IMAGE)))
    api.wait_node(server, "k3s-server")
    for name in state["workers"]:
        api.wait_node(server, name)
    wait_for_fresh_leases(state, server, started, api)
    for worker in state["workers"].values():
        api.wait_command(
            server, f"tailscale ping --until-direct=false --timeout=3s --c=1 {worker['ip']}", 90
        )
        api.wait_command(
            api.attach(worker),
            f"tailscale ping --until-direct=false --timeout=3s --c=1 {node['ip']}",
            90,
        )
    api.kube(server, "rollout", "status", "deployment/web", "--timeout=180s")
    if identities(server, api) != manifest["identities"]:
        raise RuntimeError("Kubernetes object identities or checkpoint data changed.")
    # Prove the restored API can commit new writes as well as read old state.
    api.kube(server, "create", "configmap", "after-restore", "--from-literal=result=restored")
    result = json.loads(api.kube(server, "get", "configmap", "after-restore", "-o", "json"))
    if result["data"]["result"] != "restored":
        raise RuntimeError("Post-restore API write verification failed.")
    checkpoint["phase"] = "restored"
    checkpoint["restored_sandbox_id"] = node["sandbox_id"]
    checkpoint["verified_identities"] = True
    api.save(state)
    print("PASS  New sandbox; original sandbox stopped")
    print("PASS  Same cluster, Deployment, Service, and saved data")
    print("PASS  Existing worker reconnected; API writes work", flush=True)
    if args.test:
        # Crucially, do not call deploy/test: restored objects must stand on their own.
        args.recovering = True
        api.verify(args)


def cleanup_snapshots(state, api):
    checkpoint = state.get("checkpoint", {})
    ids = set()
    if checkpoint.get("snapshot_id"):
        ids.add(checkpoint["snapshot_id"])
    # Recover a snapshot whose create response was lost before its ID was saved.
    source = checkpoint.get("manifest", {}).get("source_sandbox_id")
    if source:
        for snap in api.Sandbox.list_snapshots(source_sandbox_id=source).result():
            ids.add(snap.file_system_snapshot_id)
    for snapshot_id in ids:
        api.Sandbox.delete_snapshot(snapshot_id, missing_ok=True).result()
    if ids:
        checkpoint["deleted_snapshots"] = sorted(ids)
        api.save(state)
        print("Demo snapshots deleted.", flush=True)
