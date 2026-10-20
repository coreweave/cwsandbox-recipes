"""Disposable K3s workers, each in its own CoreWeave Kata sandbox."""

import argparse
import ipaddress
import json
import os
import re
import shlex
import sys
import time
import uuid
from pathlib import Path

from cwsandbox import FileSystemSnapshotOptions, ResourceOptions, Sandbox, SecurityContext

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".cluster-state.json"
K3S_IMAGE = "rancher/k3s:v1.37.1-k3s1"
TERMINAL = {"completed", "terminated", "failed"}


def save(state):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.chmod(0o600)
    temporary.replace(STATE)


def load():
    if not STATE.exists():
        raise RuntimeError("No cluster state; run up first or select the correct --state file.")
    return json.loads(STATE.read_text())


def attach(node):
    return Sandbox.from_id(node["sandbox_id"], data_plane_mode="gateway").result()


def execute(sb, command, timeout=180):
    result = sb.exec(["sh", "-ec", command], timeout_seconds=timeout).result()
    if result.returncode:
        # Keep failed-command details private; never copy them into a recording.
        diagnostic = STATE.with_name(".diagnostics.json")
        diagnostic.write_text(
            json.dumps(
                {
                    "command": command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }
            )
        )
        diagnostic.chmod(0o600)
        raise RuntimeError(
            f"Remote command failed ({result.returncode}); inspect private /demo logs."
        )
    return result.stdout


def wait_command(sb, command, seconds=180):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = sb.exec(["sh", "-ec", command], timeout_seconds=15).result()
        if result.returncode == 0:
            return result.stdout
        time.sleep(3)
    diagnostic = STATE.with_name(".diagnostics.json")
    diagnostic.write_text(
        json.dumps(
            {
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        )
    )
    diagnostic.chmod(0o600)
    raise TimeoutError("Remote readiness deadline exceeded; inspect private diagnostics.")


def kube(server, *args):
    return execute(server, shlex.join(["docker", "exec", "k3s", "kubectl", *args]))


def server_for(state):
    return attach(state["server"])


def new_sandbox(state, name, cpu, memory, snapshot_id=None):
    remaining = int(state["expires_at"] - time.time())
    if remaining < 180:
        raise RuntimeError("Cluster has less than three minutes left; create a fresh demo.")
    print(f"Creating {name}: {cpu} CPU / {memory} memory...", flush=True)
    sb = Sandbox.run(
        "sh",
        "/bootstrap.sh",
        container_image="docker:28.5.1-dind",
        placement_mode="serverless",
        placement_spillover="strict",
        resources=ResourceOptions(limits={"cpu": cpu, "memory": memory}),
        security_context=SecurityContext(privileged=True),
        environment_variables={
            "K3S_DISK_GIB": str(state.get("disk_gib", 12)),
            "K3S_IMAGE": state.get("k3s_image", K3S_IMAGE),
        },
        mounted_files=[
            {
                "mount_path": "/bootstrap.sh",
                "file_content": (ROOT / "bootstrap.sh").read_bytes(),
            }
        ],
        file_system_snapshot=(
            FileSystemSnapshotOptions(
                mount_path="/checkpoint", size="1Gi", file_system_snapshot_id=snapshot_id
            )
            if name == "k3s-server"
            else None
        ),
        max_lifetime_seconds=remaining,
        tags=[state["tag"]],
        data_plane_mode="gateway",
    )
    node = {"name": name, "sandbox_id": sb.sandbox_id, "cpu": cpu, "memory": memory}
    if name == "k3s-server":
        state["server"] = node
    else:
        state["workers"][name] = node
    save(state)  # Persist before any readiness/network operation can fail.
    sb.wait(timeout=180)
    wait_command(sb, "test -f /demo/ready", 240)
    node["runtime"] = sb.effective_runtime_class
    if not node["runtime"] or "kata" not in node["runtime"]:
        raise RuntimeError("This recipe requires a Kata runtime; run down to clean up.")
    save(state)
    print(f"{name}: sandbox ready ({node['runtime']})", flush=True)
    return sb, node


def setup_coordinator(server):
    execute(
        server,
        "\n".join(
            [
                "curl -fsSL --retry 3 -o /usr/local/bin/headscale "
                "https://github.com/juanfont/headscale/releases/download/v0.29.5/"
                "headscale_0.29.5_linux_amd64",
                "chmod +x /usr/local/bin/headscale",
                "mkdir -p /etc/headscale /var/lib/headscale /var/run/headscale",
            ]
        ),
    )
    for source, target in [
        ("headscale.yaml", "config.yaml"),
        ("headscale-policy.json", "policy.json"),
    ]:
        server.write_file(f"/etc/headscale/{target}", (ROOT / source).read_bytes()).result()
    execute(server, "nohup headscale serve >/demo/headscale.log 2>&1 </dev/null &")
    wait_command(server, "curl -fsS http://127.0.0.1:8080/health")
    user = json.loads(execute(server, "headscale users create demo -o json"))
    key = json.loads(
        execute(
            server,
            f"headscale preauthkeys create --user {int(user['id'])} "
            "--reusable --expiration 1h -o json",
        )
    )["key"]
    server.write_file("/demo/authkey", key.encode()).result()
    execute(
        server,
        "chmod 600 /demo/authkey; "
        "nohup tailcat --json serve --key=new --full-address 8080 "
        ">/demo/tailcat.json 2>/demo/tailcat.log </dev/null &",
    )
    wait_command(server, "test -s /demo/tailcat.json")
    print("Private network ready.", flush=True)


def connect_network(state, node, sb, server):
    key = server.read_file("/demo/authkey").result().decode().strip()
    if node["name"] != "k3s-server":
        address = json.loads(server.read_file("/demo/tailcat.json").result())["listenAddr"]
        sb.write_file("/demo/headscale.addr", address.encode()).result()
        execute(
            sb,
            "chmod 600 /demo/headscale.addr; "
            "nohup tailcat --key=new forward --bind=127.0.0.1 "
            '"$(cat /demo/headscale.addr)" 8080:8080 '
            ">/demo/tailcat.log 2>&1 </dev/null &",
        )
        wait_command(sb, "curl -fsS http://127.0.0.1:8080/health")
        sb.write_file("/demo/authkey", key.encode()).result()
    sb.write_file(
        "/demo/vpn-auth",
        (f"name=tailscale,joinKey={key},controlServerURL=http://127.0.0.1:8080").encode(),
    ).result()
    execute(
        sb,
        "chmod 600 /demo/authkey /demo/vpn-auth; "
        "tailscale up --auth-key=file:/demo/authkey "
        "--login-server=http://127.0.0.1:8080 "
        f"--hostname={shlex.quote(node['name'])} --accept-dns=false --accept-routes "
        "--timeout=60s >/demo/tailscale-up.log 2>&1",
    )
    node["ip"] = str(ipaddress.ip_address(execute(sb, "tailscale ip -4").strip()))
    peers = json.loads(execute(server, "headscale nodes list -o json"))
    peer = next(
        p for p in peers if p.get("given_name") == node["name"] or p.get("name") == node["name"]
    )
    node["network_id"] = int(peer["id"])
    save(state)
    print(f"{node['name']}: private address {node['ip']}", flush=True)


def node_command(node, server_ip=None, image=K3S_IMAGE):
    # 'host' here is the enclosing Kata guest, never the physical Kubernetes node.
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        "k3s",
        "--privileged",
        "--network",
        "host",
        "--cgroupns",
        "host",
    ]
    for mount in [
        "/var/lib/rancher/k3s/cluster:/var/lib/rancher/k3s",
        "/var/lib/rancher/k3s/kubelet:/var/lib/kubelet:rshared",
        "/var/run/tailscale:/var/run/tailscale",
        "/usr/local/bin/tailscale:/bin/tailscale:ro",
        "/demo:/demo",
        "/demo/node:/etc/rancher/node",
    ]:
        args += ["-v", mount]
    args += [
        image,
        "agent" if server_ip else "server",
        "--node-name",
        node["name"],
        "--vpn-auth-file",
        "/demo/vpn-auth",
        "--node-external-ip",
        node["ip"],
    ]
    if server_ip:
        args += ["--server", f"https://{server_ip}:6443", "--token-file", "/demo/node-token"]
    else:
        args += [
            "--disable",
            "traefik",
            "--disable",
            "servicelb",
            "--disable",
            "metrics-server",
            "--write-kubeconfig-mode",
            "600",
        ]
    return shlex.join(args) + " > /demo/k3s-container-id"


def wait_node(server, name):
    wait_command(
        server,
        shlex.join(["docker", "exec", "k3s", "kubectl", "get", "node", name]) + " >/dev/null 2>&1",
        240,
    )
    print(kube(server, "wait", "--for=condition=Ready", f"node/{name}", "--timeout=180s"), end="")


def add_worker(state, cpu, memory):
    server = server_for(state)
    name = f"k3s-worker-{state['next_worker']}"
    state["next_worker"] += 1
    save(state)
    sb, node = new_sandbox(state, name, cpu, memory)
    connect_network(state, node, sb, server)
    execute(
        server,
        "docker exec k3s cat /var/lib/rancher/k3s/server/node-token "
        "> /demo/node-token; chmod 600 /demo/node-token",
    )
    sb.write_file("/demo/node-token", server.read_file("/demo/node-token").result()).result()
    execute(
        sb,
        "chmod 600 /demo/node-token; "
        + node_command(node, state["server"]["ip"], state.get("k3s_image", K3S_IMAGE)),
    )
    wait_node(server, name)
    node["ready"] = True
    save(state)


def stop_node(node):
    sb = attach(node)
    sb.stop(graceful_shutdown_seconds=0, missing_ok=True).result()
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        status = sb.get_status().value.lower()
        if status in TERMINAL:
            node["terminal_status"] = status
            print(f"{node['name']}: sandbox stopped ({status})", flush=True)
            return
        time.sleep(2)
    raise TimeoutError(f"Stop not confirmed for {node['name']}; retry down.")


def remove_worker(state, name):
    server = server_for(state)
    node = state["workers"][name]
    if not node.get("drained"):
        print(f"Draining {name}...", flush=True)
        # Respect disruption budgets; unmanaged pods/emptyDir block removal.
        print(kube(server, "drain", name, "--ignore-daemonsets", "--timeout=120s"), end="")
        node["drained"] = True
        save(state)
    if not node.get("kubernetes_removed"):
        print(kube(server, "delete", "node", name, "--ignore-not-found"), end="")
        node["kubernetes_removed"] = True
        save(state)
    if node.get("network_id") and not node.get("network_removed"):
        execute(server, f"headscale nodes delete --identifier {int(node['network_id'])} --force")
        node["network_removed"] = True
        save(state)
    stop_node(node)
    state["removed"].append(node)
    del state["workers"][name]
    save(state)


def up(args):
    if STATE.exists():
        raise RuntimeError("Cluster state exists; use scale, or down before creating another.")
    state = {
        "tag": "k3s-demo-" + uuid.uuid4().hex[:12],
        "expires_at": time.time() + args.lifetime_seconds,
        "disk_gib": args.disk_gib,
        "k3s_image": "rancher/k3s:" + args.k3s_version,
        "workers": {},
        "removed": [],
        "next_worker": 1,
    }
    save(state)
    server, node = new_sandbox(state, "k3s-server", args.server_cpu, args.server_memory)
    setup_coordinator(server)
    connect_network(state, node, server, server)
    execute(server, node_command(node, image=state["k3s_image"]))
    wait_node(server, "k3s-server")
    print("K3s control plane ready. Adding workers...", flush=True)
    for _ in range(args.workers):
        add_worker(state, args.worker_cpu, args.worker_memory)
    status(args)


def scale(args):
    state = load()
    if any(not node.get("ready") for node in state["workers"].values()):
        raise RuntimeError("An earlier worker addition is incomplete; inspect it or run down.")
    if getattr(args, "test", False) and args.workers < len(state["workers"]):
        server = server_for(state)
        kube(server, "scale", "deployment/web", f"--replicas={1 + args.workers}")
        kube(server, "rollout", "status", "deployment/web", "--timeout=180s")
    while len(state["workers"]) < args.workers:
        add_worker(state, args.cpu, args.memory)
    while len(state["workers"]) > args.workers:
        remove_worker(state, list(state["workers"])[-1])
    status(args)
    if getattr(args, "test", False):
        test(args)


def status(args):
    state = load()
    nodes = json.loads(kube(server_for(state), "get", "nodes", "-o", "json"))
    readiness = {
        n["metadata"]["name"]: any(
            c["type"] == "Ready" and c["status"] == "True" for c in n["status"]["conditions"]
        )
        for n in nodes["items"]
    }
    print(f"{'NODE':<18} {'CPU':<6} {'MEMORY':<8} {'PRIVATE IP':<14} STATUS")
    for node in [state["server"], *state["workers"].values()]:
        phase = "Ready" if readiness.get(node["name"]) else "NotReady"
        print(
            f"{node['name']:<18} {node['cpu']:<6} {node['memory']:<8} "
            f"{node.get('ip', 'pending'):<14} {phase}"
        )
        if getattr(args, "ids", False):
            print(f"  sandbox: {node['sandbox_id']}")


def deploy(args):
    state = load()
    server = server_for(state)
    server.write_file("/demo/workloads.yaml", (ROOT / "workloads.yaml").read_bytes()).result()
    output = kube(server, "apply", "-f", "/demo/workloads.yaml")
    output += kube(server, "scale", "deployment/web", f"--replicas={1 + len(state['workers'])}")
    output += kube(server, "rollout", "status", "deployment/web", "--timeout=180s")
    output += kube(
        server,
        "get",
        "pods",
        "-l",
        "app=web",
        "-o",
        "custom-columns=NAME:.metadata.name,STATUS:.status.phase,"
        "IP:.status.podIP,NODE:.spec.nodeName",
    )
    if not getattr(args, "quiet", False):
        print(output, end="")


def test(args):
    args.quiet = True
    print("Deploying nginx and testing the cluster...", flush=True)
    deploy(args)
    verify(args)


def ready_nodes(nodes, expected):
    items = nodes["items"]
    return {n["metadata"]["name"] for n in items} == expected and all(
        any(c["type"] == "Ready" and c["status"] == "True" for c in n["status"]["conditions"])
        for n in items
    )


def verify(args):
    def fetch(server, *command):
        if getattr(args, "recovering", False):
            return wait_command(
                server, shlex.join(["docker", "exec", "k3s", "kubectl", *command]), 90
            )
        return kube(server, *command)

    state = load()
    server = server_for(state)
    expected = {"k3s-server", *state["workers"]}
    nodes = json.loads(kube(server, "get", "nodes", "-o", "json"))
    if not ready_nodes(nodes, expected):
        raise RuntimeError("Expected exactly the tracked nodes, all Ready.")
    pods = json.loads(kube(server, "get", "pods", "-l", "app=web", "-o", "json"))
    web = {p["spec"]["nodeName"]: p for p in pods["items"] if p["status"]["phase"] == "Running"}
    if set(web) != expected:
        raise RuntimeError("Expected one running web replica per node; run deploy first.")
    clients, checks = [], []
    try:
        for node_name in sorted(expected):
            name = "check-" + node_name
            spec = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": name},
                "spec": {
                    "nodeSelector": {"kubernetes.io/hostname": node_name},
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 1,
                    "containers": [
                        {
                            "name": "client",
                            "image": "busybox:1.37",
                            "command": ["sleep", "300"],
                            "resources": {
                                "requests": {"cpu": "20m", "memory": "16Mi"},
                                "limits": {"cpu": "100m", "memory": "64Mi"},
                            },
                        }
                    ],
                },
            }
            server.write_file("/demo/check.json", json.dumps(spec).encode()).result()
            kube(server, "apply", "-f", "/demo/check.json")
            clients.append(name)
            kube(server, "wait", "--for=condition=Ready", f"pod/{name}", "--timeout=120s")
        for worker in state["workers"]:
            for source, destination in [("k3s-server", worker), (worker, "k3s-server")]:
                ip = str(ipaddress.ip_address(web[destination]["status"]["podIP"]))
                response = fetch(
                    server,
                    "exec",
                    "check-" + source,
                    "--",
                    "wget",
                    "-T",
                    "15",
                    "-qO-",
                    "http://" + ip,
                ).strip()
                if response != f"Hello from {destination}":
                    raise RuntimeError("Cross-node HTTP response did not match destination.")
                checks.append(
                    {
                        "source": source,
                        "destination": destination,
                        "pod_ip": ip,
                        "response": response,
                    }
                )
        for node_name in sorted(expected):
            response = fetch(
                server,
                "exec",
                "check-" + node_name,
                "--",
                "wget",
                "-T",
                "15",
                "-qO-",
                "http://web.default.svc.cluster.local",
            )
            if not response.startswith("Hello from k3s-"):
                raise RuntimeError("Service DNS / HTTP check failed.")
        log_node = next(iter(state["workers"]), "k3s-server")
        logs = kube(server, "logs", web[log_node]["metadata"]["name"], "--tail=5")
        if "GET /" not in logs:
            raise RuntimeError("Expected nginx request logs from the remote node.")
        result = kube(server, "exec", "check-" + log_node, "--", "echo", "Remote exec: OK")
        if result.strip() != "Remote exec: OK":
            raise RuntimeError("Remote exec check failed.")
        evidence = {
            "passed": True,
            "nodes": nodes,
            "web_pods": pods,
            "cross_node": checks,
            "logs": logs,
            "state": state,
            "verified_at": time.time(),
        }
        STATE.with_name(".verification.json").write_text(json.dumps(evidence, indent=2))
        print(f"PASS  {len(expected)} Ready nodes, nginx on every node")
        print(f"PASS  Cross-node HTTP ({len(checks)} checks)")
        print(f"PASS  Service DNS ({len(expected)} nodes)")
        print("PASS  Remote logs and exec", flush=True)
    finally:
        if clients:
            kube(server, "delete", "pod", *clients, "--ignore-not-found", "--wait=true")


def down(args):
    state = load()
    failures = []
    for node in [*reversed(list(state["workers"].values())), state.get("server")]:
        if not node:
            continue
        try:
            stop_node(node)
        except Exception as exc:
            failures.append(f"{node['name']}: {type(exc).__name__}")
        save(state)
    if failures:
        raise RuntimeError("Cleanup incomplete; retry down: " + "; ".join(failures))
    import recovery

    recovery.cleanup_snapshots(state, sys.modules[__name__])
    STATE.replace(STATE.with_name(".cluster-last-run.json"))
    print("All tracked sandboxes stopped. Private network removed.", flush=True)


def cpu_quantity(value):
    if not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?|[0-9]+m)", value):
        raise argparse.ArgumentTypeError(
            "CPU must be positive cores (2, 0.5) or millicores (500m)."
        )
    number = float(value[:-1]) / 1000 if value.endswith("m") else float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("CPU must be greater than zero.")
    return value


def memory_quantity(value):
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:Ki|Mi|Gi|Ti|K|M|G|T)", value):
        raise argparse.ArgumentTypeError(
            "Memory must be a positive quantity such as 4096Mi or 8Gi."
        )
    if float(re.match(r"[0-9.]+", value)[0]) <= 0:
        raise argparse.ArgumentTypeError("Memory must be greater than zero.")
    return value


def worker_count(value):
    count = int(value)
    if not 0 <= count <= 10:
        raise argparse.ArgumentTypeError("This demo supports 0 through 10 workers.")
    return count


def k3s_version(value):
    if not re.fullmatch(r"v\d+\.\d+\.\d+-k3s\d+", value):
        raise argparse.ArgumentTypeError("Expected a pinned version such as v1.35.1-k3s1")
    return value


def main():
    global STATE
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE)
    sub = parser.add_subparsers(dest="action", required=True)
    launch = sub.add_parser("up")
    launch.add_argument("--workers", type=worker_count, default=1)
    launch.add_argument("--cpu", "--worker-cpu", dest="worker_cpu", type=cpu_quantity, default="2")
    launch.add_argument(
        "--memory", "--worker-memory", dest="worker_memory", type=memory_quantity, default="4Gi"
    )
    launch.add_argument("--server-cpu", type=cpu_quantity, default="4")
    launch.add_argument("--server-memory", type=memory_quantity, default="8Gi")
    launch.add_argument("--k3s-version", type=k3s_version, default=K3S_IMAGE.split(":")[1])
    launch.add_argument(
        "--lifetime-seconds",
        type=int,
        choices=range(600, 14401),
        default=3600,
        metavar="600..14400",
        help="Shared cluster deadline (default: 1 hour)",
    )
    launch.add_argument(
        "--disk-gib",
        type=int,
        choices=range(12, 129),
        default=12,
        metavar="12..128",
        help="Sparse ext4 disk size per node",
    )
    resize = sub.add_parser("scale")
    resize.add_argument("--workers", type=worker_count, required=True)
    resize.add_argument("--cpu", type=cpu_quantity, default="2", help="CPU for new workers only")
    resize.add_argument("--memory", type=memory_quantity, default="4Gi", help="New workers only")
    resize.add_argument(
        "--test",
        action="store_true",
        help="Resize the sample nginx deployment and run cluster tests",
    )
    sub.add_parser("status").add_argument("--ids", action="store_true")
    for action in ["deploy", "verify", "test", "down", "resume", "id", "tag"]:
        sub.add_parser(action)
    sub.add_parser("snapshot").add_argument(
        "--prepare-only",
        action="store_true",
        help="Stage a consistent checkpoint for cwsandbox snapshots create",
    )
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--test", action="store_true")
    restore_parser.add_argument("--snapshot", help="Snapshot ID created by the standard CLI")
    sub.add_parser("kubectl").add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    STATE = args.state.resolve()
    try:
        if args.action in {"id", "tag"}:
            state = load()
            print(state["server"]["sandbox_id"] if args.action == "id" else state["tag"])
        elif args.action in {"snapshot", "restore", "resume"}:
            import recovery

            getattr(recovery, args.action)(args, sys.modules[__name__])
        elif args.action == "kubectl":
            print(kube(server_for(load()), *args.arguments), end="")
        else:
            globals()[args.action](args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        print(
            "State is retained. Inspect private logs or run: ./sandbox-k3s down",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
