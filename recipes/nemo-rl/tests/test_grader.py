import pytest
from nemo_rl_sandbox.grader import grade, score_response


def test_final_answer_and_brevity():
    assert score_response("2 + 2 = 4", "4") > 1
    assert 0 < score_response("5", "4") < 1
    assert score_response("4", "4") > score_response("The answer is 4", "4")
    assert score_response("4.5", "4") < 1
    assert score_response("The answer is 4.", "4") > 1


def test_batch_length_is_checked():
    with pytest.raises(ValueError):
        grade({"responses": ["4"], "expected": []})


def test_generated_code_is_only_text():
    assert score_response('__import__("os").system("exit 1")', "4") < 1
