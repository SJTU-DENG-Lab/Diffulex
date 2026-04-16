import pickle
from types import SimpleNamespace

import pytest

from diffulex.engine.status import DllmBlockStatus, DllmBlockType
from diffulex.strategy.d2f.engine.request import D2fReq
from diffulex.strategy.multi_bd.engine.request import MultiBDReq


def _multi_block_config():
    return SimpleNamespace(
        block_size=32,
        buffer_size=4,
        mask_token_id=-1,
        decoding_thresholds=SimpleNamespace(
            add_block_threshold=0.1,
            semi_complete_threshold=0.9,
            accept_threshold=0.9,
        ),
        eos=-1,
        max_model_len=2048,
    )


@pytest.mark.parametrize("req_cls", [MultiBDReq, D2fReq])
def test_req_pickle_roundtrip_restores_runtime_refs(req_cls) -> None:
    req = req_cls(list(range(96)))
    req.page_size = 32
    req.init_multi_block(_multi_block_config())
    req.status_history.append(req.status)
    req.dllm_blocks[0].status = DllmBlockStatus.IN_CACHE
    req.dllm_blocks[0].block_type = DllmBlockType.OUT_OF_CONTEXT
    req.dllm_blocks[-1].block_type = DllmBlockType.LAST_IN_CONTEXT

    restored = pickle.loads(pickle.dumps(req))

    assert restored.req_id == req.req_id
    assert restored.token_ids == req.token_ids
    assert restored.status_history == req.status_history
    assert restored.dllm_block_buffer.req is restored
    assert all(block.req is restored for block in restored.dllm_blocks)
    assert restored.dllm_blocks[0].status == DllmBlockStatus.IN_CACHE
    assert restored.dllm_blocks[0].block_type == DllmBlockType.OUT_OF_CONTEXT
    assert restored.dllm_blocks[-1].block_type == DllmBlockType.LAST_IN_CONTEXT

    buffer_block_ids = {id(block) for block in restored.dllm_block_buffer.dllm_blocks}
    for block in restored.dllm_blocks:
        if id(block) in buffer_block_ids:
            assert block.dllm_block_buffer is restored.dllm_block_buffer
        else:
            assert block.dllm_block_buffer is None
