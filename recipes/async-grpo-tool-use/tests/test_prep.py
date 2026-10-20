from async_grpo_tool_use import dapo_data


def _dapo(question, answer):
    return {"prompt": [{"role": "user", "content": question}], "reward_model": {"ground_truth": answer}}


def test_dapo_rows_are_unique_gym_tasks():
    rows = [_dapo("q1", "1"), _dapo("q1", "1"), _dapo("q2", 2), _dapo("q3", "3")]
    out = list(dapo_data.unique_rows(rows, limit=2))
    assert [r["question"] for r in out] == ["q1", "q2"]
    assert out[1]["expected_answer"] == "2"
    assert out[0]["agent_ref"]["name"] == "math_tir_agent" and out[0]["split"] == "train"


def test_eval_rows_are_marked_for_the_weave_evaluation():
    [row] = dapo_data.unique_rows([_dapo("q1", "1")], agent=dapo_data.EVAL_AGENT)
    assert row["agent_ref"]["name"] == "math_eval_agent" and row["split"] == "eval"
