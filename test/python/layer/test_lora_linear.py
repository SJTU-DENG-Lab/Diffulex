import copy

import torch

from diffulex.layer.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from diffulex.utils import parallelism


def _mock_tp(monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    parallelism.reset_model_parallelism_metadata()
    monkeypatch.setattr(
        parallelism,
        "_MODEL_PARALLELISM_METADATA",
        parallelism.ModelParallelismMetadata.from_world(
            tp_size=1,
            ep_size=1,
            world_size=1,
            global_rank=0,
        ),
    )


def test_replicated_linear_lora_forward_uses_non_transposed_weights(monkeypatch) -> None:
    _mock_tp(monkeypatch)
    layer = ReplicatedLinear(input_size=4, output_size=3, r=2, lora_alpha=2.0, lora_dropout=0.0)
    layer.weight.data.zero_()
    layer.lora_A.data.copy_(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
    )
    layer.lora_B.data.copy_(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
    )

    x = torch.tensor([[2.0, 3.0, 0.0, 0.0]])
    out = layer(x)

    expected = torch.tensor([[2.0, 3.0, 5.0]])
    torch.testing.assert_close(out, expected)


def _set_lora_state(layer, weight, lora_a, lora_b) -> None:
    layer.weight.data.copy_(weight)
    layer.lora_A.data.copy_(lora_a)
    layer.lora_B.data.copy_(lora_b)


def _assert_merge_matches_unmerged(layer, x: torch.Tensor) -> None:
    unmerged = copy.deepcopy(layer).eval()
    merged = copy.deepcopy(layer).eval()
    merged.merge_lora()
    torch.testing.assert_close(merged(x), unmerged(x), atol=1e-5, rtol=1e-5)


def test_replicated_linear_merge_matches_unmerged(monkeypatch) -> None:
    _mock_tp(monkeypatch)
    layer = ReplicatedLinear(input_size=4, output_size=3, r=2, lora_alpha=4.0, lora_dropout=0.0)
    _set_lora_state(
        layer,
        weight=torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
            ]
        ),
        lora_a=torch.tensor(
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
            ]
        ),
        lora_b=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    _assert_merge_matches_unmerged(layer, x)


def test_column_parallel_linear_merge_matches_unmerged(monkeypatch) -> None:
    _mock_tp(monkeypatch)
    layer = ColumnParallelLinear(input_size=4, output_size=6, r=2, lora_alpha=2.0, lora_dropout=0.0)
    _set_lora_state(
        layer,
        weight=torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
                [1.7, 1.8, 1.9, 2.0],
                [2.1, 2.2, 2.3, 2.4],
            ]
        ),
        lora_a=torch.tensor(
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
            ]
        ),
        lora_b=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
            ]
        ),
    )
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    _assert_merge_matches_unmerged(layer, x)


def test_row_parallel_linear_merge_matches_unmerged(monkeypatch) -> None:
    _mock_tp(monkeypatch)
    layer = RowParallelLinear(input_size=4, output_size=3, r=2, lora_alpha=2.0, lora_dropout=0.0)
    _set_lora_state(
        layer,
        weight=torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
            ]
        ),
        lora_a=torch.tensor(
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
            ]
        ),
        lora_b=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    _assert_merge_matches_unmerged(layer, x)
