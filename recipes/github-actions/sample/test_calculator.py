import os
import unittest

from calculator import add


class CalculatorTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_orchestrator_secrets_are_absent(self):
        for name in ("CWSANDBOX_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN"):
            self.assertNotIn(name, os.environ)
