"""The recipe's ``.env`` file, read by every command: ``async-grpo``, ``launch`` and ``cleanup``."""

from __future__ import annotations

import os
import re
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
INLINE_COMMENT = re.compile(r"\s+#.*$")


def read_dotenv(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines of a .env file. Comments, blank lines, ``export`` and quotes are
    allowed; an unquoted value ends at `` #``, so ``SAMPLER_REPLICAS=2  # comment`` is ``2``."""
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if value[:1] in ("'", '"') and value[0] in value[1:]:
            value = value[1 : value.index(value[0], 1)]
        else:
            value = INLINE_COMMENT.sub("", value)
        values[key.strip()] = value
    return values


def load() -> None:
    """Add the recipe's .env settings to the environment. Values already in the environment win."""
    for key, value in read_dotenv(ENV_FILE).items():
        os.environ.setdefault(key, value)
