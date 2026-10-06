"""Stop one known sandbox using explicit CoreWeave authentication."""

import argparse

from cwsandbox import AuthStrategy, Sandbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sandbox_id")
    args = parser.parse_args()
    sandbox = Sandbox.from_id(
        args.sandbox_id, auth=AuthStrategy.COREWEAVE_API_KEY
    ).result(timeout=60)
    sandbox.stop(missing_ok=True).result(timeout=60)
    print(f"Sandbox {args.sandbox_id} stopped.")


if __name__ == "__main__":
    main()
