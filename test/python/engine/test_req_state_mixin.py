import pickle
from types import SimpleNamespace

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


class MarkerStateMixin:
    def _customize_req_state(self, state):
        state = super()._customize_req_state(state)
        state.pop("runtime_only", None)
        return state

    def _restore_req_runtime_state(self):
        super()._restore_req_runtime_state()
        self.runtime_only = f"restored:{self.req_id}"


class ExtendedMultiBDReq(MarkerStateMixin, MultiBDReq):
    pass


def test_req_state_mixin_chains_additional_mixins() -> None:
    req = ExtendedMultiBDReq(list(range(96)))
    req.page_size = 32
    req.init_multi_block(_multi_block_config())
    req.runtime_only = "live"

    restored = pickle.loads(pickle.dumps(req))

    assert restored.runtime_only == f"restored:{req.req_id}"
    assert restored.dllm_block_buffer.req is restored
    assert all(block.req is restored for block in restored.dllm_blocks)
