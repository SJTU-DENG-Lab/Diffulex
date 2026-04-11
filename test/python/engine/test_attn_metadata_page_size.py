import pytest

from types import SimpleNamespace

from diffulex.attention.metadata import AttnMetaDataBase
from diffulex.mixin.multi_block.engine.model_runner import ModelRunnerMultiBlockMixin


class _Runner(ModelRunnerMultiBlockMixin):
    def __init__(self):
        self.page_size = 32
        self.block_size = 32
        self.rank = 0
        self.is_prefix_full = True
        self.config = SimpleNamespace(buffer_size=4, kv_cache_layout="unified")
        self.captured_kwargs = None

    def prepare_page_tables(self, reqs):
        return SimpleNamespace()

    def set_attn_metadata(self, **kwargs):
        self.captured_kwargs = kwargs

    def fetch_attn_metadata(self):
        return SimpleNamespace(init_multi_block=lambda **kwargs: None)


class _Req:
    def __init__(self):
        self.status = SimpleNamespace(PENDING="PENDING")
        self.is_prefilling = True
        self.block_size = 32
        self.buffer_size = 4
        self.running_sequence = list(range(32, 64))
        self.running_position_ids = list(range(32, 64))
        self.in_cache_len = 32
        self.running_len = 64
        self.valid_len = 32
        self.cache_len = 64
        self.to_cache_len = 32
        self.prefix_len = 64
        self.padded_prefix_len = 64
        self.page_table = [0]
        self.page_cache_missed = [True]
        self.num_cached_tokens = 32
        self.dllm_blocks = [
            SimpleNamespace(start=0, end=32, rel_page_id=0, is_to_cache=False),
            SimpleNamespace(start=32, end=64, rel_page_id=0, is_to_cache=True),
        ]

    def step(self):
        return None


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
def test_prepare_chunked_prefill_passes_runtime_page_size_to_attn_metadata() -> None:
    runner = _Runner()
    req = _Req()

    runner.prepare_chunked_prefill_multi_block([req])

    assert runner.captured_kwargs is not None
    assert runner.captured_kwargs["page_size"] == 32
    assert runner.captured_kwargs["block_size"] == 32


def test_multi_block_metadata_accepts_smaller_block_than_page() -> None:
    metadata = AttnMetaDataBase(page_size=8, block_size=4)

    metadata.init_multi_block()

    assert metadata.page_size == 8
    assert metadata.block_size == 4


def test_multi_block_metadata_rejects_block_larger_than_page() -> None:
    metadata = AttnMetaDataBase(page_size=4, block_size=8)

    with pytest.raises(ValueError, match="block_size 8 must be <= page_size 4"):
        metadata.init_multi_block()
