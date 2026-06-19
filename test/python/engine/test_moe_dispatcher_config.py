from __future__ import annotations

from types import SimpleNamespace

import pytest

from diffulex.config import Config


def test_standard_moe_backend_rejects_requested_ep_size(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="MoE A2A backends are currently unsupported"):
        Config(
            model=str(model_dir),
            hf_config=SimpleNamespace(max_position_embeddings=2048),
            tensor_parallel_size=2,
            expert_parallel_size=8,
            data_parallel_size=1,
            moe_dispatcher_backend="standard",
            device_ids=[0, 1],
        )
