# Contributing a recipe

Start from [`_template/`](_template/): copy it to `recipes/<kebab-case-name>/` and fill it in. Name the directory after the workload stack (for example `taubench-verl`), not after marketing terms.

## Checklist

Every recipe must:

- Be self-contained under `recipes/<name>/`: its own `pyproject.toml`, lockfile, config, and tests. A user who copies the directory out of this repo can still run it.
- Demonstrate one workload. Split unrelated ideas into separate recipes.
- Have a README that takes a new user from zero to a working run: what it demonstrates, architecture, prerequisites, setup, numbered run steps with expected output, where to see results, and explicit cleanup for every billable resource.
- Ship a `.env.example` with placeholders only. One inline comment per variable: what it is and where to get it. Mark required vs optional. All secrets come from `.env` or a secret store, never from source files.
- Pin dependency versions (lockfile plus explicit pins for framework versions the code depends on).
- Include smoke tests that run without provisioning cloud resources.
- State GPU/quota requirements and rough cost up front.
- Use copy-pasteable commands. If a step can confuse someone, add the one sentence that removes the confusion.

## Style

Plain, direct English. Short sentences. No filler. Comments only where the why is not obvious from the code. Run the formatter, linter, and tests before opening a PR.

Do not commit `.env`, logs, checkpoints, or local planning documents.

## Contributor License Agreement

Contributors must agree to the [CoreWeave CLA](./CLA.md) when pushing code to this project.

Agreement with the CoreWeave CLA must be signified by including a `Signed-off-by`
trailer in every submitted Git commit to this repository. By signing off, you certify that you have the right to submit the contribution and that you agree to and are bound by the CoreWeave Contributor License Agreement in effect at the date of your submission, found in [`CLA.md`](./CLA.md) in the root of this repository, which governs your submission. If you are contributing on behalf of an entity, you further certify that you are authorized to bind that entity to the CLA.

Sign each commit with the `--signoff` (`-s`) option to [`git commit`](https://git-scm.com/docs/git-commit#Documentation/git-commit.txt---signoff). Git has no configuration option that adds the trailer automatically; if you want it on every commit, use an alias such as `git config alias.ci "commit -s"` or a `prepare-commit-msg` hook.

## Licensing

This project is licensed under Apache-2.0 (see [`LICENSE`](./LICENSE)) and follows the [REUSE](https://reuse.software/) specification. REUSE requires the license text in [`LICENSES/Apache-2.0.txt`](./LICENSES/Apache-2.0.txt). Licensing metadata lives in [`REUSE.toml`](./REUSE.toml): its aggregate annotation covers every file by default, so new files need no SPDX header. If you add material under a different license or copyright, declare it with an inline SPDX header or a `REUSE.toml` annotation and include any additional license text in `LICENSES/<SPDX-License-Identifier>.txt`. Run `reuse lint` from the repository root before opening a PR.
