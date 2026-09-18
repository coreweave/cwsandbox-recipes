import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("verify", Path(__file__).parents[1] / "verify.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.events = []
        self.turns = [
            {
                "id": "root-1",
                "session_id": "session-test",
                "subagent_id": None,
                "status": "completed",
            },
            {
                "id": "root-2",
                "session_id": "session-test",
                "subagent_id": None,
                "status": "completed",
            },
        ]
        self.items = []
        for index, (name, metrics) in enumerate(verify.expected_metrics().items()):
            (self.output / f"{name}.json").write_text(json.dumps(metrics))
            (self.output / f"{name}.py").write_text("import csv\nprint('test artifact')\n")
            self.events.append(
                {
                    "received_at": "2026-01-15T15:00:00Z",
                    "event": {
                        "type": "agent.session.subagent.created",
                        "subagent": {
                            "id": f"child-{name}",
                            "session_id": "session-test",
                            "instructions": [{"type": "input_text", "text": f"SPECIALIST={name}"}],
                        },
                    },
                }
            )
            self.turns.append(
                {
                    "id": f"turn-{name}",
                    "session_id": "session-test",
                    "subagent_id": f"child-{name}",
                    "status": "completed",
                    "started_at": 100 + index,
                    "completed_at": 110 + index,
                }
            )
            self.items.append(
                {
                    "id": f"command-{name}",
                    "type": "command_execution",
                    "turn_id": f"turn-{name}",
                    "status": "completed",
                    "exit_code": 0,
                    "command": f"python3 output/{name}.py",
                }
            )
        (self.output / "incident.md").write_text(
            "Synthetic incident in west following deploy-b. Correlation does not prove causation. "
            "Sources: latency.json, errors.json, capacity.json, fixtures/deployments.json."
        )
        (self.output / "followup.md").write_text(
            "Synthetic follow-up using incident.md and latency.json. "
            "Correlation does not prove causation."
        )

    def check(self, **kwargs):
        return verify.verify_output(
            self.output,
            events=self.events,
            turns=self.turns,
            items=self.items,
            expected_session_id="session-test",
            **kwargs,
        )

    def test_fixture_ground_truth(self):
        metrics = verify.expected_metrics()
        west = next(
            group
            for group in metrics["latency"]["groups"]
            if group["window"] == "after" and group["region"] == "west"
        )
        self.assertEqual(
            west, {"window": "after", "region": "west", "count": 30, "p50_ms": 400, "p95_ms": 1200}
        )
        errors = next(
            group
            for group in metrics["errors"]["groups"]
            if group["window"] == "after" and group["region"] == "west"
        )
        self.assertEqual(errors["error_count"], 6)
        self.assertEqual(errors["error_rate"], 0.2)
        capacity = next(
            group
            for group in metrics["capacity"]["groups"]
            if group["window"] == "after" and group["region"] == "west"
        )
        self.assertEqual(capacity["mean_utilization_pct"], 90)
        self.assertEqual(capacity["max_queue_depth"], 18)

    def test_complete_evidence_and_outputs(self):
        result = self.check(require_overlap=True)
        self.assertTrue(result["verified"])
        self.assertTrue(result["delegation_verified"])
        self.assertTrue(result["overlap"]["established"])
        self.assertEqual(len(result["specialist_subagent_ids"]), 3)

    def test_live_projection_without_instructions_or_exit_codes(self):
        for event in self.events:
            event["event"]["subagent"]["instructions"] = None
        for item in self.items:
            item["exit_code"] = None
        result = self.check(require_overlap=True)
        self.assertTrue(result["delegation_verified"])
        self.assertFalse(result["exit_code_evidence"]["all_specialists_have_zero_exit"])
        self.assertEqual(len(result["exit_code_evidence"]["missing_for_command_ids"]), 3)
        with self.assertRaisesRegex(verify.VerificationError, "Zero exit codes could not"):
            self.check(require_overlap=True, require_exit_codes=True)

    def test_encrypted_instructions_do_not_replace_command_evidence(self):
        for event in self.events:
            event["event"]["subagent"]["instructions"] = [
                {"type": "encrypted_content", "encrypted_content": "test-opaque"}
            ]
        self.assertTrue(self.check(require_exit_codes=True)["verified"])
        self.items.clear()
        with self.assertRaisesRegex(verify.VerificationError, "No completed Python execution"):
            self.check()

    def test_failed_status_with_missing_exit_code_does_not_count(self):
        self.items[0].update(status="failed", exit_code=None)
        with self.assertRaises(verify.VerificationError):
            self.check()

    def test_boolean_exit_code_does_not_count_as_zero(self):
        self.items[0]["exit_code"] = False
        with self.assertRaises(verify.VerificationError):
            self.check()

    def test_one_child_cannot_execute_all_three_roles(self):
        for turn in self.turns[2:]:
            turn["subagent_id"] = "child-latency"
        with self.assertRaisesRegex(verify.VerificationError, "three distinct"):
            self.check()

    def test_two_children_claiming_one_script_are_ambiguous(self):
        extra = copy.deepcopy(self.items[0])
        extra.update(id="duplicate-command", turn_id="turn-errors")
        self.items.append(extra)
        with self.assertRaisesRegex(verify.VerificationError, "attribution is ambiguous"):
            self.check()

    def test_wrong_metric(self):
        path = self.output / "latency.json"
        data = json.loads(path.read_text())
        data["groups"][0]["p95_ms"] += 1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(verify.VerificationError, "Wrong numeric metric"):
            self.check()

    def test_nonfinite_metric(self):
        path = self.output / "latency.json"
        data = json.loads(path.read_text())
        data["groups"][0]["p95_ms"] = float("nan")
        path.write_text(json.dumps(data))
        with self.assertRaises(verify.VerificationError):
            self.check()

    def test_malformed_json(self):
        (self.output / "errors.json").write_text("{bad json")
        with self.assertRaisesRegex(verify.VerificationError, "Cannot read JSON"):
            self.check()

    def test_missing_artifact(self):
        (self.output / "capacity.py").unlink()
        with self.assertRaisesRegex(verify.VerificationError, "Missing artifact"):
            self.check()

    def test_invalid_python(self):
        (self.output / "capacity.py").write_text("this is not Python !")
        with self.assertRaisesRegex(verify.VerificationError, "Invalid Python"):
            self.check()

    def test_created_children_without_actual_work_fails(self):
        self.items.clear()
        with self.assertRaisesRegex(verify.VerificationError, "No completed Python execution"):
            self.check()

    def test_root_commands_cannot_impersonate_subagent_work(self):
        for item in self.items:
            item["turn_id"] = "root-1"
        with self.assertRaisesRegex(verify.VerificationError, "No completed Python execution"):
            self.check()

    def test_failed_command_does_not_count(self):
        self.items[0]["exit_code"] = 1
        with self.assertRaises(verify.VerificationError):
            self.check()

    def test_printed_or_commented_command_does_not_count(self):
        for command in (
            "echo 'python3 output/latency.py'",
            "cat input.txt # python3 output/latency.py",
            "false; python3 output/latency.py; true",
        ):
            with self.subTest(command=command):
                self.items[0]["command"] = command
                with self.assertRaises(verify.VerificationError):
                    self.check()

    def test_shell_wrapper_for_exact_invocation(self):
        self.items[0]["command"] = "/bin/bash -lc 'python3 output/latency.py'"
        self.assertTrue(self.check()["verified"])

    def test_three_records_of_one_child_do_not_count(self):
        for event in self.events:
            event["event"]["subagent"]["id"] = "same-child"
        with self.assertRaisesRegex(verify.VerificationError, "three distinct"):
            self.check()

    def test_evidence_cannot_mix_sessions(self):
        self.events[0]["event"]["subagent"]["session_id"] = "different-session"
        with self.assertRaisesRegex(verify.VerificationError, "different API sessions"):
            self.check()

    def test_followup_requires_second_root_turn(self):
        self.turns.pop(0)
        with self.assertRaisesRegex(verify.VerificationError, "two completed root turns"):
            self.check()

    def test_created_times_and_receive_times_do_not_prove_overlap(self):
        for turn in self.turns:
            turn.pop("started_at", None)
            turn["created_at"] = 100
        result = self.check()
        self.assertFalse(result["overlap"]["established"])
        with self.assertRaisesRegex(verify.VerificationError, "overlap could not"):
            self.check(require_overlap=True)

    def test_adjacent_intervals_do_not_prove_overlap(self):
        for index, turn in enumerate(self.turns[2:]):
            turn.update(started_at=index * 10, completed_at=(index + 1) * 10)
        self.assertFalse(self.check()["overlap"]["established"])

    def test_repeated_turn_records_cannot_prove_overlap(self):
        for index, turn in enumerate(self.turns[2:]):
            turn.update(started_at=index * 10, completed_at=(index + 1) * 10)
        self.turns.append(copy.deepcopy(self.turns[2]))
        self.assertFalse(self.check()["overlap"]["established"])

    def test_report_must_reference_evidence(self):
        (self.output / "incident.md").write_text("Looks fine.")
        with self.assertRaisesRegex(verify.VerificationError, "must cite"):
            self.check()

    def test_cli_rejects_failed_or_unidentified_run(self):
        (self.output / "events.jsonl").write_text("")
        for run in ({"status": "failed", "session_id": "session-test"}, {"status": "completed"}):
            with self.subTest(run=run):
                (self.output / "run.json").write_text(json.dumps(run))
                stream = io.StringIO()
                with (
                    patch("sys.argv", ["verify.py", "--output-dir", str(self.output)]),
                    patch("sys.stdout", stream),
                ):
                    self.assertEqual(verify.main(), 1)
                self.assertFalse(json.loads(stream.getvalue())["verified"])


if __name__ == "__main__":
    unittest.main()
