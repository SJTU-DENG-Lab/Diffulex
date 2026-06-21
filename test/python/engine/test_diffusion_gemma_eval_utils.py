from diffulex_bench.tasks.gsm8k.diffusion_gemma_utils import process_results_math


def test_diffusion_gemma_gsm8k_accepts_boxed_numeric_answer_with_units() -> None:
    result = process_results_math(
        {"ground_truth_answer": "48"},
        ["The answer is \\boxed{48g}."],
    )

    assert result == {"exact_match": 1}
