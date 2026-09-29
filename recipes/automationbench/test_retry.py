import unittest

from harbor.agents.installed.base import ApiRateLimitError, ApiUsageLimitError
from harbor.trial.queue import TrialQueue

from adapter import terminal_quota_error
from run import retry_policy


class RetryTests(unittest.TestCase):
    def test_quota_and_transient_errors_retry(self):
        queue = TrialQueue(4, retry_policy())
        for name in (
            "SandboxResourceExhaustedError",
            "SandboxUnavailableError",
            "ApiUsageLimitError",
            "ApiRateLimitError",
        ):
            self.assertTrue(queue._should_retry_exception(name))

    def test_model_failures_do_not_retry(self):
        queue = TrialQueue(4, retry_policy())
        for name in (
            "AgentSafetyRefusalError",
            "AgentTimeoutError",
            "RuntimeError",
            "BadRequestError",
        ):
            self.assertFalse(queue._should_retry_exception(name))

    def test_backoff_is_bounded(self):
        queue = TrialQueue(4, retry_policy())
        self.assertEqual(
            [queue._calculate_backoff_delay_sec(n) for n in range(5)],
            [30, 60, 120, 240, 300],
        )

    def test_terminal_quota_is_classified(self):
        self.assertIs(
            terminal_quota_error(
                "Aborted rollout due to RateLimitError: Error code: 429"
            ),
            ApiRateLimitError,
        )
        self.assertIs(
            terminal_quota_error(
                "Aborted rollout due to RateLimitError: insufficient_quota"
            ),
            ApiUsageLimitError,
        )
        self.assertIs(
            terminal_quota_error("Traceback\nopenai.RateLimitError: Error code: 429"),
            ApiRateLimitError,
        )

    def test_successful_request_retry_and_model_errors_do_not_rerun_trial(self):
        for log in (
            "[retry] RateLimitError: 429; request later succeeded",
            "Tool returned 429",
            "Aborted rollout due to BadRequestError: must be a valid JSON object string",
        ):
            self.assertIsNone(terminal_quota_error(log))


class RecipeTests(unittest.TestCase):
    def test_task_manifests_match_expected_coverage(self):
        from pathlib import Path

        root = Path(__file__).parent
        scored = (root / "tasks-scored.txt").read_text().splitlines()
        simple = (root / "tasks-simple.txt").read_text().splitlines()
        self.assertEqual(len(set(scored)), 600)
        self.assertEqual(len(set(simple)), 200)
        self.assertTrue(all(name.startswith("simple.") for name in simple))
        self.assertFalse(set(scored) & set(simple))
        for domain in ["sales", "marketing", "operations", "support", "finance", "hr"]:
            self.assertEqual(sum(n.startswith(domain + ".") for n in scored), 100)

    def test_task_file_job_configuration_and_credentials(self):
        import asyncio
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import AsyncMock, Mock, patch

        import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "tasks.txt"
            selection.write_text(
                "sales.multi_hop_lookup\n\nsimple.email_sf_contact_phone_update\n"
            )
            env = {
                "CWSANDBOX_API_KEY": "sandbox-test-secret",
                "MODEL_API_KEY": "model-test-secret",
                "MODEL_NAME": "example",
                "MODEL_BASE_URL": "https://example.invalid/v1",
            }
            result = Mock()
            result.stats.n_errored_trials = 0
            result.stats.n_completed_trials = 2
            result.model_dump_json.return_value = "{}"
            job = Mock()
            job.run = AsyncMock(return_value=result)
            with (
                patch.dict(os.environ, env, clear=True),
                patch(
                    "sys.argv",
                    [
                        "run.py",
                        "--tasks-file",
                        str(selection),
                        "--output",
                        str(root / "out"),
                        "--concurrency",
                        "2",
                    ],
                ),
                patch.object(run.Job, "create", AsyncMock(return_value=job)) as create,
            ):
                asyncio.run(run.main())
                config = create.call_args.args[0]
                self.assertEqual(len(config.tasks), 2)
                self.assertEqual(config.n_concurrent_trials, 2)
                raw = (root / "out/job-config.json").read_text()
                self.assertNotIn(env["CWSANDBOX_API_KEY"], raw)
                self.assertNotIn(env["MODEL_API_KEY"], raw)

    def test_cleanup_only_recorded_ids_and_reports_failures(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch

        from cwsandbox import AuthStrategy

        import cleanup

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sandbox-ids.jsonl").write_text(
                '{"sandbox_id":"owned-a"}\n{"sandbox_id":"owned-a"}\n'
            )
            sandbox = Mock(status="completed")
            with (
                patch.object(cleanup.Sandbox, "delete") as delete,
                patch.object(cleanup.Sandbox, "from_id") as lookup,
            ):
                lookup.return_value.result.return_value = sandbox
                cleanup.cleanup(root)
                delete.assert_called_once_with(
                    "owned-a", auth=AuthStrategy.COREWEAVE_API_KEY, missing_ok=True
                )
                sandbox.status = "running"
                with self.assertRaises(SystemExit):
                    cleanup.cleanup(root)


if __name__ == "__main__":
    unittest.main()
