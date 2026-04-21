import pytest
import torch
import torch.nn.functional as F

from diffulex.moe.topk import NaiveTopKRouter, build_topk_router


@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_naive_topk_router_matches_torch(scoring_func: str, renormalize: bool):
    router_logits = torch.tensor(
        [
            [0.2, -1.0, 3.0, 0.5, -0.2],
            [-2.0, 1.5, 1.4, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    router = NaiveTopKRouter(top_k=3, renormalize=renormalize, scoring_func=scoring_func)

    output = router(router_logits)

    if scoring_func == "softmax":
        scores = F.softmax(router_logits, dim=-1, dtype=torch.float)
    else:
        scores = torch.sigmoid(router_logits.float())
    expected_weights, expected_ids = torch.topk(scores, 3, dim=-1, sorted=False)
    if renormalize:
        expected_weights = expected_weights / (expected_weights.sum(dim=-1, keepdim=True) + 1e-20)

    assert output.router_logits is router_logits
    assert output.ids.dtype == torch.int32
    assert torch.equal(output.ids, expected_ids.to(torch.int32))
    assert torch.allclose(output.weights, expected_weights)


def test_build_topk_router_keeps_legacy_impl_names_on_naive_router():
    for impl in ("naive", "triton"):
        assert isinstance(build_topk_router(impl, top_k=2), NaiveTopKRouter)
