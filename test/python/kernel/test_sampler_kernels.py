import pytest
import torch

from diffulex.sampler.llada2 import _llada2_greedy_sample_eager
from diffulex_kernel.python.sampler_kernels import greedy_confidence


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_greedy_confidence_matches_llada2_eager_with_vocab_and_forbidden_mask() -> None:
    torch.manual_seed(0)
    logits = torch.randn((7, 97), device="cuda", dtype=torch.bfloat16)
    logits[:, 89:] = 4.0
    logits[0, 3] = 8.0
    logits[0, 11] = 8.0
    logits[1, 5] = 9.0

    confidence, sampled_tokens, initial_confidence = greedy_confidence(
        logits,
        vocab_limit=89,
        forbidden_token_id=5,
    )
    expected_confidence, expected_tokens, expected_initial_confidence = _llada2_greedy_sample_eager(
        logits,
        tokenizer_vocab_size=89,
        mask_token_id=5,
        sanitize_logits=False,
    )

    torch.testing.assert_close(sampled_tokens, expected_tokens)
    torch.testing.assert_close(confidence, expected_confidence, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(initial_confidence, expected_initial_confidence, rtol=2e-3, atol=2e-4)
    assert int(sampled_tokens[0].item()) == 11
    assert int(sampled_tokens[1].item()) != 5
