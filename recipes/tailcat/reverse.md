# Call a sandbox from CKS

Complete steps 1–2 in [README.md](README.md) first. This example uses the same image and dedicated namespace. The sandbox serves HTTP only on loopback port `8081`, with Tailcat providing access from the CKS caller.

```text
CKS Job → localhost:18081 → Tailcat tunnel → sandbox localhost:8081
```

## 1. Start the sandbox service

In terminal A, from this recipe directory:

```bash
uv run --frozen --env-file .env sandbox.py serve
```

Expected output:

```text
Sandbox: <sandbox-id>
Address saved privately to .tailcat/sandbox.addr
Run the CKS caller in another terminal. Lifetime: at most 30 minutes.
Press Enter or Ctrl-C to stop the sandbox and revoke its address.
```

Leave this terminal running. The script creates the address file with mode `0600` and refuses to overwrite an existing one. If a previous run crashed, stop its sandbox before removing the stale file and retrying.

## 2. Give the CKS caller the address

In terminal B, from the same directory:

```bash
set -a
source .env
set +a

kubectl --context "$KUBE_CONTEXT" -n tailcat-demo \
  create secret generic sandbox-tailcat \
  --from-file=address=.tailcat/sandbox.addr --dry-run=client -o yaml \
  | kubectl --context "$KUBE_CONTEXT" -n tailcat-demo apply --server-side -f -
```

The secret value travels through a pipe, not a terminal or a command argument. Restrict access to Secrets and pod exec in this namespace: either permission can expose the connection credential.

## 3. Run the caller

```bash
kubectl --context "$KUBE_CONTEXT" -n tailcat-demo \
  delete job cks-caller --ignore-not-found --wait=true
envsubst '${TAILCAT_IMAGE}' < k8s/caller.yaml \
  | kubectl --context "$KUBE_CONTEXT" apply -f -
kubectl --context "$KUBE_CONTEXT" -n tailcat-demo \
  wait --for=condition=complete job/cks-caller --timeout=180s
kubectl --context "$KUBE_CONTEXT" -n tailcat-demo logs job/cks-caller
```

Expected response (initial connection retries may also appear):

```text
hello from sandbox
```

The Job reads the address from a mounted Secret, starts a loopback Tailcat forwarder, and calls `http://127.0.0.1:18081/`. Tailcat diagnostics remain in a private file inside the pod. The Job exits after the request and has a three-minute deadline.

## Optional: CKS → sandbox → CKS

This variant uses two tunnels. The CKS Job calls the sandbox's `/roundtrip` endpoint, which then calls the private CKS Service through the forward tunnel:

```text
CKS Job → sandbox HTTP /roundtrip → private cks-api Service
```

1. Press Enter in terminal A to stop the previous sandbox. Complete README step 3 to get the CKS bridge's current address, then start the new sandbox:

   ```bash
   uv run --frozen --env-file .env sandbox.py roundtrip \
     --cks-address .tailcat/cks.addr
   ```

   It first prints `hello from CKS`, then waits for the caller.

2. Repeat step 2 above in terminal B to replace the Secret with the **new** sandbox address. Delete the previous caller Job, then set its request path to `/roundtrip` before applying:

   ```bash
   kubectl --context "$KUBE_CONTEXT" -n tailcat-demo \
     delete job cks-caller --ignore-not-found --wait=true
   kubectl set env --local -f k8s/caller.yaml REQUEST_PATH=/roundtrip -o yaml \
     | envsubst '${TAILCAT_IMAGE}' \
     | kubectl --context "$KUBE_CONTEXT" apply -f -
   kubectl --context "$KUBE_CONTEXT" -n tailcat-demo \
     wait --for=condition=complete job/cks-caller --timeout=180s
   kubectl --context "$KUBE_CONTEXT" -n tailcat-demo logs job/cks-caller
   ```

   Expected response:

   ```text
   hello from sandbox
   upstream: hello from CKS
   ```

## Cleanup

Press Enter in terminal A to stop the sandbox, then follow the [README cleanup steps](README.md#cleanup) to delete the CKS namespace, including the address Secret and caller Job, remove local addresses, and remove the registry image when finished.
