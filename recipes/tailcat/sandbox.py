"""Run from your workstation; the SDK creates and cleans up one sandbox."""

import argparse
import json
import os
from pathlib import Path

from cwsandbox import Sandbox


def execute(sandbox: Sandbox, command: list[str], timeout: int = 120) -> str:
    result = sandbox.exec(command, timeout_seconds=timeout).result()
    if result.returncode != 0:
        # Tailcat diagnostics may contain the bearer address. Keep them private.
        raise RuntimeError(
            f"Sandbox command failed (exit {result.returncode}). "
            "See the troubleshooting section in README.md."
        )
    return result.stdout


def write_private(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Refuse to overwrite a previous run's address or follow a symlink.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(value + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fetch", "serve", "roundtrip"])
    parser.add_argument("--image", default=os.environ.get("TAILCAT_IMAGE"))
    parser.add_argument("--cks-address", type=Path)
    parser.add_argument("--address-out", type=Path, default=Path(".tailcat/sandbox.addr"))
    args = parser.parse_args()
    if not args.image:
        parser.error("set TAILCAT_IMAGE or pass --image")
    if args.mode != "serve" and not args.cks_address:
        parser.error("fetch and roundtrip require --cks-address")
    if args.mode != "fetch" and args.address_out.exists():
        parser.error("--address-out already exists; choose a fresh path")
    cks_address = args.cks_address.read_text().strip() if args.cks_address else None
    if cks_address is not None and not cks_address.startswith("tc"):
        parser.error("--cks-address must contain a Tailcat address")

    address_written = False
    try:
        with Sandbox.run(
            "sleep",
            "1800",
            container_image=args.image,
            placement_mode="serverless",
            placement_spillover="strict",
            max_lifetime_seconds=1800,
            resources={"cpu": "1", "memory": "1Gi"},
            tags=["tailcat-recipe"],
            # No services / public endpoints are needed for this tunnel.
        ) as sandbox:
            sandbox.wait(timeout=180)
            print(f"Sandbox: {sandbox.sandbox_id}", flush=True)
            execute(sandbox, ["sh", "-ec", "umask 077; mkdir /tmp/tailcat"])

            if cks_address:
                # Transfer through the authenticated file API, not an exec argv.
                sandbox.write_file("/tmp/tailcat/cks.addr", cks_address.encode()).result()
                execute(
                    sandbox,
                    [
                        "sh",
                        "-ec",
                        """
                    umask 077
                    chmod 600 /tmp/tailcat/cks.addr
                    nohup tailcat --key=new forward --bind=127.0.0.1 \
                      "$(cat /tmp/tailcat/cks.addr)" 18080:18080 \
                      >/tmp/tailcat/forward.log 2>&1 </dev/null &
                """,
                    ],
                )
                output = execute(
                    sandbox,
                    [
                        "curl",
                        "--fail",
                        "--silent",
                        "--show-error",
                        "--retry",
                        "20",
                        "--retry-all-errors",
                        "--retry-delay",
                        "1",
                        "--retry-max-time",
                        "90",
                        "--max-time",
                        "5",
                        "http://127.0.0.1:18080/",
                    ],
                )
                print(output, end="", flush=True)

            if args.mode == "fetch":
                return

            execute(
                sandbox,
                [
                    "sh",
                    "-ec",
                    """
                umask 077
                nohup python /opt/recipes/sandbox_service.py \
                  >/tmp/tailcat/app.log 2>&1 </dev/null &
                nohup tailcat --json serve --key=new --full-address 8081 \
                  >/tmp/tailcat/address.json 2>/tmp/tailcat/serve.log </dev/null &
                for attempt in $(seq 1 60); do
                  if test -s /tmp/tailcat/address.json && \
                    curl -fsS --max-time 1 http://127.0.0.1:8081/ >/dev/null; then
                    exit 0
                  fi
                  sleep 1
                done
                exit 1
            """,
                ],
            )
            metadata = json.loads(sandbox.read_file("/tmp/tailcat/address.json").result())
            write_private(args.address_out, metadata["listenAddr"])
            address_written = True
            print(f"Address saved privately to {args.address_out}", flush=True)
            print(
                "Run the CKS caller in another terminal. Lifetime: at most 30 minutes.", flush=True
            )
            input("Press Enter or Ctrl-C to stop the sandbox and revoke its address. ")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if address_written:
            args.address_out.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
