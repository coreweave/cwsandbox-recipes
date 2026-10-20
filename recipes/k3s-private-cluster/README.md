# A private Kubernetes cluster across sandboxes

Create a K3s control plane and independently sized worker nodes in CoreWeave serverless sandboxes. Add worker sandboxes, run workloads across them, then drain and remove workers when they are no longer needed.

**Resources:** CPU only. Defaults are one 4-CPU/8-GiB server and one 2-CPU/4-GiB worker: 6 CPUs and 12 GiB total. Each additional worker defaults to 2 CPUs/4 GiB. The server also has a 1-GiB filesystem-snapshot staging mount. Snapshot storage is retained until cleanup. Every sandbox expires at approximately the same one-hour cluster deadline. An unchanged default run therefore requests at most 6 vCPU-hours and 12 GiB-hours. Scaling to two workers with a new 4-CPU/8-GiB worker raises the total to 10 CPUs and 20 GiB. Actual charges depend on your rates and runtime. Stop the cluster when finished.

## What this demonstrates

- One Kubernetes cluster spanning separate Kata guest VMs.
- A private Tailscale network with a disposable Headscale coordinator, reached through Tailcat.
- Worker scale-up with configurable CPU/memory; existing workers retain their sizes.
- Scale-down that drains Kubernetes workloads, removes the node/network identity, and confirms sandbox termination.
- Live checks for pod traffic in both directions, Service DNS, remote logs, and exec.
- Control-plane recovery into a new sandbox, preserving cluster identity and reconnecting the existing workers.

## Architecture

```text
Your workstation: cwsandbox SDK provisions and controls the demo

Server sandbox (Kata VM)               Worker sandbox(s) (separate Kata VMs)
  Docker -> K3s server                   Docker -> K3s agent
  Tailscale <------- encrypted node + pod networking ------> Tailscale
  Headscale on localhost:8080 <---- Tailcat forwarding ----> localhost:8080
```

Tailcat bootstraps access to the coordinator; Tailscale carries Kubernetes traffic. K3s's experimental VPN integration advertises each node's pod subnet. No Tailscale account or pre-existing tailnet is required. No public sandbox endpoints are created. Both sides need outbound access to download dependencies and reach the public Tailcat/Tailscale DERP infrastructure. Connections may be relayed; encrypted private connectivity does not mean a dedicated private physical network.

The coordinator is inside the server sandbox and exists only for this cluster. Its policy permits communication among this run's enrolled nodes and automatically approves this cluster's `10.42.0.0/16` pod routes. Do not reuse that policy in an unrelated/shared tailnet.

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/).
- A [CoreWeave Sandbox API token](https://docs.coreweave.com/products/sandboxes/get-started) with serverless capacity, filesystem snapshots enabled, and policy permission for privileged Kata workloads.
- An amd64 Kata runner whose guest supports TUN, overlayfs, loop devices, and cgroup v2. The recipe was tested on `kata-qemu`, guest kernel 6.18.12.
- Enough quota for the server plus your desired worker count. No GPUs or nested virtualization extensions are needed.

Tools installed inside the sandboxes: Docker 28.5.1, K3s v1.37.1+k3s1, Tailscale 1.104.1, Headscale 0.29.5, and Tailcat 0.7.0. You do not need local Docker or kubectl.

## Setup

```bash
git clone https://github.com/coreweave/cwsandbox-recipes.git
cd cwsandbox-recipes/recipes/k3s-private-cluster
cp .env.example .env
# Fill in CWSANDBOX_API_KEY in .env.
uv sync --frozen
```

The executable `sandbox-k3s` wrapper uses uv and reads an optional adjacent `.env` as data. It also accepts the same credentials through environment variables. The API token remains on your workstation. Network enrollment credentials and the Kubernetes join token are exchanged using the authenticated Sandbox SDK file API, stored privately inside the guests, and never printed by the recipe. Do not publish private `/demo` logs or network address files.

If you already cloned the repository, start in `recipes/k3s-private-cluster`. Run the commands below from that directory. Use `./sandbox-k3s --help` to list commands and `./sandbox-k3s <command> --help` for options.

## Run

1. Create a server and one worker:

   ```bash
   ./sandbox-k3s up \
     --workers 1 --cpu 2 --memory 4Gi
   ```

   Both nodes should report `Ready`. The status table includes resources, private IPs, and node readiness. Use `status --ids` to include sandbox IDs. Server sizing can also be set with `--server-cpu` and `--server-memory` on `up`.

2. Deploy nginx on every node and verify connectivity:

   ```bash
   ./sandbox-k3s test
   ```

   The Deployment uses required pod anti-affinity so there is one replica per node. `test` combines `deploy` and `verify`. `verify` creates temporary client pods on each node, checks cross-node HTTP in both directions, checks Service DNS, fetches remote logs, tests exec, and removes its clients. Expect `PASS  2 Ready nodes, nginx on every node`. Evidence is saved to `.verification.json` next to the selected state file.

3. Add a larger worker sandbox:

   ```bash
   ./sandbox-k3s scale --workers 2 --cpu 4 --memory 8Gi --test
   ```

   `--workers` is the desired **worker** count; it excludes the control plane. This adds one 4-CPU/8-GiB sandbox. The original worker remains 2 CPUs/4 GiB. `--test` updates the sample Deployment to three replicas and runs the checks; expect a successful three-node verification.

   CPU supports cores (`2`, `0.5`) or millicores (`2000m`). Memory supports quantities such as `4096Mi` and `8Gi`. Use at least 2 CPUs/4 GiB per worker for this demo. The platform applies its own quota/policy limits. Resource flags apply only to newly created workers; they do not resize existing sandboxes. The CLI table shows requested sandbox limits. Kubernetes may report larger guest capacity because Kata includes VM overhead; this demo does not tune kubelet reservations for production scheduling.

4. Remove the newest worker:

   ```bash
   ./sandbox-k3s scale --workers 1 --test
   ```

   The explicit `--test` flag adjusts the sample nginx workload because its required anti-affinity allows only one replica per node, then runs the checks again. Without `--test`, `scale` changes infrastructure only; manage your own workload replica count separately. The scale command drains the newest worker, deletes its Kubernetes Node, revokes its Headscale identity, stops its sandbox, and waits for a terminal status. The remaining cluster should pass its checks again.

   Drain respects disruption budgets and does not force removal of unmanaged pods or discard `emptyDir` data. If drain fails, the sandbox remains; resolve the workload restriction and retry. A failed drain may leave the node cordoned. Use `./sandbox-k3s kubectl uncordon <node>` if you abandon removal. Scale-down progress is recorded so completed steps can be skipped on retry.

5. Inspect or run other Kubernetes commands:

   ```bash
   ./sandbox-k3s status
   ./sandbox-k3s kubectl get pods -A
   ./sandbox-k3s kubectl logs deployment/web --tail=5
   ```

   These invoke kubectl through the authenticated Sandbox SDK. The Kubernetes API remains private.

## Inspect with the standard cwsandbox CLI

The pinned dependency includes the CLI extra. With credentials already in your environment:

```bash
source .venv/bin/activate
cwsandbox ls --tag "$(./sandbox-k3s tag)"
cwsandbox get "$(./sandbox-k3s id)"
cwsandbox exec --data-plane-mode gateway "$(./sandbox-k3s id)" kubectl get nodes
```

If credentials are only in `.env`, prefix standard CLI commands with `uv run --env-file .env`, for example `uv run --env-file .env cwsandbox ls --tag "$(./sandbox-k3s tag)"`. The recipe wrapper already reads `.env`. `id` returns the current control-plane sandbox ID, including after a restore. The guest's `kubectl` wrapper runs the binary inside its K3s container.

## Snapshot and restore the control plane

Start with a running cluster and sample workloads (`up`, then `test`). The short recipe flow is:

```bash
./sandbox-k3s snapshot
./sandbox-k3s restore --test
```

To show the platform snapshot operation with the standard CLI, use this equivalent flow:

```bash
./sandbox-k3s snapshot --prepare-only
cwsandbox snapshots create "$(./sandbox-k3s id)" | tee .snapshot-id
./sandbox-k3s restore --snapshot "$(cat .snapshot-id)" --test
```

Use `set -o pipefail` in Bash/Zsh when piping CLI output. The standard CLI needs the environment setup above, or the `uv run --env-file .env` prefix. Run only one of these flows per cluster.

Preparation creates a unique ConfigMap, records Kubernetes object UIDs, stops K3s, briefly stops the private-network daemons, and copies a consistent checkpoint into `/checkpoint`. It saves the K3s SQLite database, server token and certificates, node password, Headscale database and key, and Tailscale identity. The network daemons then resume, while K3s stays paused. `cwsandbox snapshots create` captures those staged files using the platform filesystem-snapshot API. The snapshot contains credentials; keep it private.

Restore confirms the snapshot is ready and belongs to this prepared source, stops the original sandbox, confirms termination, and creates a new Kata sandbox from the snapshot. It rebuilds Docker/K3s, restores the database and private IP, and refreshes each worker's Tailcat connection to the restored coordinator. Existing worker sandboxes remain running. Stopping the original first prevents two control planes from sharing the same identities.

The proof checks all of these without reapplying `workloads.yaml`:

- A new control-plane sandbox ID and confirmed termination of the old sandbox.
- The same default namespace, nginx Deployment, Service, and ConfigMap UIDs, with the original ConfigMap value.
- Fresh kubelet lease renewals after restore, private-network reachability in both directions, and a successful rollout of the original Deployment. Historical Ready conditions in a snapshot are not sufficient.
- A new API write, followed by cross-node HTTP, Service DNS, remote logs, and exec when `--test` is specified. HTTP checks allow up to 90 seconds for recovery before failing.

This is a **cold control-plane restore**, with API downtime. It restores application definitions and cluster state; it does not capture VM memory, running processes, worker disks, persistent-volume data, or arbitrary application storage. Workers must still exist, membership must match the checkpoint, and at least five minutes must remain before the shared deadline. New changes after the checkpoint are rolled back. This demo supports one checkpoint/restore cycle per run.

If you decide not to restore after preparation, run `./sandbox-k3s resume` to start the original K3s control plane again. A failed backup may also leave K3s paused; `resume` or `down` is the recovery path. Failed remote-command details are saved privately in `.diagnostics.json`; do not publish that file or private guest logs. An interrupted restore retains the new sandbox ID for cleanup; use `down` before starting a fresh run.

## Cleanup and recovery

```bash
./sandbox-k3s down
```

`down` stops all tracked worker sandboxes and then the server, verifies termination, deletes this run's control-plane snapshots, and saves `.cluster-last-run.json`. This also discovers snapshots created through the standard CLI for the checkpoint source. It does not preserve the checkpoint for future use. It destroys the demo rather than draining it. If a stop fails, it still attempts the other stops, retains state, and exits nonzero; retry `down`. Deleting the server also removes Headscale and the Tailcat listener. No external tailnet nodes or public endpoints remain to clean up.

Keep `.cluster-state.json` until cleanup completes. It contains resource IDs, not join secrets. An interrupted `up` or `scale` preserves successfully created sandbox IDs for `down`. The unique run tag can identify resources in the Sandbox API if the state file is lost. All sandboxes also have the one-hour deadline as a backstop. Worker additions inherit the remaining cluster lifetime, not a fresh hour.

Run one command at a time per state file. For independent demos use separate recipe directories or `./sandbox-k3s --state /path/to/private-state.json <command>`; use the same state path for all subsequent commands.

## Limits

This is an ephemeral demonstration, not a production Kubernetes service: one control plane, no HA, no durable application storage, no autoscaler, and no in-place resizing. Scaling is explicitly requested through the CLI, with a demo limit of ten workers. K3s's VPN integration and Tailcat are experimental; public Tailcat relays provide no SLA. This recipe is not a throughput or startup-time benchmark.

Kata contains privilege within the guest. Nested Docker uses that guest's network and cgroups; it does not request physical-host namespaces from the platform. K3s runs inside an ext4-backed Docker container because kubelet could not account for the outer virtiofs root. The 12-GiB sparse ext4 filesystem is a working storage arrangement, not a platform-enforced tenant disk quota.

## Local checks

```bash
uv run --frozen pytest tests -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
sh -n bootstrap.sh sandbox-k3s
```

Sources: [K3s backup and restore](https://docs.k3s.io/datastore/backup-restore), [K3s Tailscale integration](https://docs.k3s.io/networking/distributed-multicloud), [Tailcat](https://github.com/tailscale/tailcat/tree/v0.7.0), [Headscale](https://headscale.net/), and the adjacent [`tailcat` recipe](../tailcat/).

## Development sizing and version

The default shared deadline remains one hour. For a larger disposable dev
environment, `up --lifetime-seconds 14400 --disk-gib 80` allows up to four hours
and configures an 80 GiB sparse ext4 backing file on each node. Workers inherit
the remaining deadline and disk size. Disk allocation is still ephemeral and
limited by the enclosing sandbox's available storage; this is not a persistent
volume or a storage reservation. The platform enforces the deadline.

Pin another Kubernetes version with `--k3s-version v1.35.1-k3s1`. The chosen
image is saved with the cluster and reused for new workers and restoration.
This does not upgrade or downgrade an existing cluster.
