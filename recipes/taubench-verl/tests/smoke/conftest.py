"""Shared test setup.

tau-bench calls load_dotenv() when first imported, which injects the
developer's local .env into os.environ mid-suite and makes tests
order-dependent. Import it up front and drop whatever it loaded so every
test sees a hermetic environment.
"""

import os

_before = set(os.environ)
try:
    import tau_bench  # noqa: F401
except ImportError:
    pass
for _key in set(os.environ) - _before:
    del os.environ[_key]
