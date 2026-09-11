"""CoreWeave sandbox plane for τ-bench environments.

Modules:

* :mod:`verl_taubench.sandbox.env_server` -- the τ-bench env server that runs
  *inside* a sandbox. stdlib + tau_bench only; uploaded verbatim.
* :mod:`verl_taubench.sandbox.client` -- async client + pluggable transport.
* :mod:`verl_taubench.sandbox.pool` -- warm pool with lease/reset/return.
* :mod:`verl_taubench.sandbox.simulator` -- the remote user simulator, which
  deliberately runs outside the sandbox.
* :mod:`verl_taubench.sandbox.trainer` -- GPU training job launcher that fires
  ``train_grpo.sh`` as the sandbox main process via ``cwsandbox``.
"""
