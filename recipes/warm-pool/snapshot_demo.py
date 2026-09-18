"""Verify that a clean filesystem snapshot can seed independent sandboxes."""

from uuid import uuid4

from cwsandbox import AuthStrategy, FileSystemSnapshotOptions, Sandbox, SandboxDefaults

from warm_pool import LIFETIME_SECONDS

AUTH = AuthStrategy.COREWEAVE_API_KEY  # Use AuthStrategy.WANDB for W&B credentials.


def main():
    tag = f"warm-pool-snapshot-{uuid4().hex[:12]}"
    print(f"Run tag: {tag}", flush=True)
    defaults = SandboxDefaults(
        auth=AUTH,
        container_image="python:3.11-slim",
        placement_mode="serverless",
        resources={"cpu": "2", "memory": "4Gi"},
        max_lifetime_seconds=LIFETIME_SECONDS,
        tags=(tag,),
    )
    snapshot_id = None
    try:
        with Sandbox.run(
            defaults=defaults,
            file_system_snapshot=FileSystemSnapshotOptions(mount_path="/work", size="1Gi"),
        ) as seed:
            seed.exec(["sh", "-c", "echo clean > /work/baseline.txt"], check=True).result()
            snapshot_id = seed.snapshot().result()  # Waits for READY.
            print(f"Temporary snapshot: {snapshot_id}", flush=True)
        for _ in range(2):
            with Sandbox.run(
                defaults=defaults,
                file_system_snapshot=FileSystemSnapshotOptions(
                    mount_path="/work", size="1Gi", file_system_snapshot_id=snapshot_id
                ),
            ) as fork:
                result = fork.exec(
                    [
                        "sh",
                        "-c",
                        'test "$(cat /work/baseline.txt)" = clean && '
                        "test ! -e /work/private.txt && "
                        "echo changed > /work/baseline.txt && "
                        "echo private > /work/private.txt && echo 'independent restore verified'",
                    ],
                    check=True,
                ).result()
                print(result.stdout.strip(), flush=True)
    finally:
        if snapshot_id:
            Sandbox.delete_snapshot(snapshot_id, missing_ok=True, auth=AUTH).result()
    print("Seed and restored sandboxes stopped; snapshot deleted.")


if __name__ == "__main__":
    main()
