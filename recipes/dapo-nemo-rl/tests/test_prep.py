from dapo_nemo_rl import dapo_data


def _dapo(question, answer):
    return {"prompt": [{"role": "user", "content": question}], "reward_model": {"ground_truth": answer}}


def test_dapo_rows_are_unique_gym_tasks():
    rows = [_dapo("q1", "1"), _dapo("q1", "1"), _dapo("q2", 2), _dapo("q3", "3")]
    out = list(dapo_data.unique_rows(rows, limit=2))
    assert [r["question"] for r in out] == ["q1", "q2"]
    assert out[1]["expected_answer"] == "2"
    assert out[0]["agent_ref"]["name"] == "math_with_judge_simple_agent"
    assert out[0]["responses_create_params"]["input"] == [{"role": "user", "content": "q1"}]
