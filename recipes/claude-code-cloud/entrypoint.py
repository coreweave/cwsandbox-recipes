"""Forward shutdown signals and prevent automatic retries after a process error."""

import os
import signal
import subprocess
import sys


def main():
    process = subprocess.Popen(sys.argv[1:], start_new_session=True)

    def shutdown(signum, frame):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    code = process.wait()
    # Work orders are single-use. A nonzero container exit can trigger a platform retry.
    print(f"Recipe process exited with code {code}; workload will not retry", flush=True)


if __name__ == "__main__":
    main()
