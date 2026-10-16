from dapo_nemo_rl import calibrate


def test_cases_cover_both_formats_and_wrong_answers():
    cases = calibrate.build_cases(["34", "\\frac{1}{2}"])
    assert [c["kind"] for c in cases[:4]] == ["boxed_correct", "answer_correct", "boxed_wrong", "answer_wrong"]
    assert cases[0]["response"].endswith("\\boxed{34}.") and cases[1]["response"].endswith("Answer: 34")
    assert cases[2]["response"].endswith("\\boxed{35}.")
    assert all(c["expected"] == "\\frac{1}{2}" for c in cases[4:])


def test_summary_counts_agreement_per_kind():
    cases = calibrate.build_cases(["1", "2"])
    # The verifier misses the "Answer:" format and grades everything else right.
    results = [{"reward": 0.0 if c["kind"] == "answer_correct" else calibrate.KINDS[c["kind"]]} for c in cases]
    report = calibrate.summarize(cases, results)
    assert report["boxed_correct"] == {"agree": 1.0, "cases": 2}
    assert report["answer_correct"]["agree"] == 0.0
