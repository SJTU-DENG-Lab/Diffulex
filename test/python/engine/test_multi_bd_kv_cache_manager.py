from types import SimpleNamespace

from diffulex.engine.status import DllmBlockStatus
from diffulex.engine.dllm_block import DllmBlock
from diffulex.strategy.multi_bd.engine.kv_cache_manager import MultiBDKVCacheManager
from diffulex.strategy.multi_bd.engine.request import MultiBDReq


class FakeReq:
    def __init__(self, pages: list[list[int]], *, cache_len: int, to_cache_len: int):
        self._pages = pages
        self.page_table = [0, 1]
        self.cache_len = cache_len
        self.to_cache_len = to_cache_len

    @property
    def num_pages(self) -> int:
        return len(self._pages)

    def num_pages_with_seq_len(self, seq_len: int) -> int:
        return (seq_len + len(self._pages[0]) - 1) // len(self._pages[0])

    def page(self, index: int) -> list[int]:
        return list(self._pages[index])


def _build_manager() -> MultiBDKVCacheManager:
    config = SimpleNamespace(num_pages=16, page_size=32)
    return MultiBDKVCacheManager(config)


def _seed_prefix_pages(manager: MultiBDKVCacheManager, req: FakeReq) -> None:
    for rel_page_id, page_id in enumerate(req.page_table):
        page = manager._allocate_page(page_id)
        prefix = manager.pages[req.page_table[rel_page_id - 1]].hash if rel_page_id > 0 else -1
        token_ids = req.page(rel_page_id)
        h = manager.compute_hash(token_ids, prefix)
        page.update(h, token_ids)
        manager.hash_to_page_id[h] = page_id


def test_may_append_finalizes_intermediate_page_when_appending_multiple_pages() -> None:
    pages = [
        [1000 + i for i in range(32)],
        [2000 + i for i in range(32)],
        [3000 + i for i in range(32)],
        [4000 + i for i in range(32)],
        [5000 + i for i in range(32)],
    ]
    req = FakeReq(pages, cache_len=128, to_cache_len=64)
    manager = _build_manager()
    _seed_prefix_pages(manager, req)

    manager.may_append(req)

    assert req.page_table == [0, 1, 2, 3]

    page2 = manager.pages[2]
    page3 = manager.pages[3]
    expected_h2 = manager.compute_hash(req.page(2), manager.pages[1].hash)

    assert page2.hash == expected_h2
    assert page2.token_ids == req.page(2)
    assert manager.hash_to_page_id[expected_h2] == 2
    assert page3.hash == -1


def test_may_append_uses_last_page_table_entry_when_finalizing_previous_append() -> None:
    pages = [
        [1000 + i for i in range(32)],
        [2000 + i for i in range(32)],
        [3000 + i for i in range(32)],
        [4000 + i for i in range(32)],
        [5000 + i for i in range(32)],
    ]
    req = FakeReq(pages, cache_len=128, to_cache_len=64)
    manager = _build_manager()
    _seed_prefix_pages(manager, req)

    manager.may_append(req)

    req.cache_len = 160
    req.to_cache_len = 32
    manager.may_append(req)

    assert req.page_table == [0, 1, 2, 3, 4]

    page3 = manager.pages[3]
    page4 = manager.pages[4]
    expected_h3 = manager.compute_hash(req.page(3), manager.pages[2].hash)

    assert page3.hash == expected_h3
    assert page3.token_ids == req.page(3)
    assert manager.hash_to_page_id[expected_h3] == 3
    assert page4.hash == -1


def test_may_append_does_not_allocate_extra_page_when_cache_already_fully_covered() -> None:
    pages = [
        [1000 + i for i in range(32)],
        [2000 + i for i in range(32)],
    ]
    req = FakeReq(pages, cache_len=64, to_cache_len=32)
    manager = _build_manager()
    manager.free_page_ids.clear()

    manager.may_append(req)

    assert req.page_table == [0, 1]


def test_can_append_uses_missing_pages_not_token_bytes() -> None:
    pages = [
        [1000 + i for i in range(32)],
        [2000 + i for i in range(32)],
        [3000 + i for i in range(32)],
    ]
    req = FakeReq(pages, cache_len=96, to_cache_len=32)
    manager = _build_manager()
    manager.free_page_ids.clear()

    assert manager.can_append(req) is False

    manager.free_page_ids.append(5)
    assert manager.can_append(req) is True


def test_apply_cached_prefix_pages_marks_prefix_blocks_in_cache() -> None:
    config = SimpleNamespace(
        block_size=32,
        buffer_size=4,
        mask_token_id=-1,
        decoding_thresholds=SimpleNamespace(
            add_block_threshold=0.1,
            semi_complete_threshold=0.9,
            decoding_threshold=0.9,
        ),
        eos=-1,
        max_model_len=2048,
    )
    manager = _build_manager()

    token_ids = list(range(64))
    req1 = MultiBDReq(token_ids)
    req1.page_size = 32
    req1.init_multi_block(config)
    manager.allocate(req1)
    manager.free(req1)

    req2 = MultiBDReq(token_ids)
    req2.page_size = 32
    req2.init_multi_block(config)
    manager.allocate(req2)
    req2.apply_cached_prefix_pages()

    assert req2.num_cached_tokens == 64
    assert req2.in_cache_len == 64
    assert req2.page_cache_missed == [False, False]
    assert all(req2.dllm_blocks[i].status == DllmBlockStatus.IN_CACHE for i in range(2))


def test_apply_cached_prefix_page_marks_all_blocks_inside_cached_page() -> None:
    config = SimpleNamespace(
        block_size=4,
        buffer_size=4,
        mask_token_id=-1,
        decoding_thresholds=SimpleNamespace(
            add_block_threshold=0.1,
            semi_complete_threshold=0.9,
            decoding_threshold=0.9,
        ),
        eos=-1,
        max_model_len=2048,
    )
    manager = MultiBDKVCacheManager(SimpleNamespace(num_pages=16, page_size=8))

    token_ids = list(range(16))
    req1 = MultiBDReq(token_ids)
    req1.page_size = 8
    req1.init_multi_block(config)
    manager.allocate(req1)
    manager.free(req1)

    req2 = MultiBDReq(token_ids)
    req2.page_size = 8
    req2.init_multi_block(config)
    manager.allocate(req2)
    req2.apply_cached_prefix_pages()

    assert req2.num_cached_tokens == 16
    assert req2.in_cache_len == 16
    assert req2.page_cache_missed == [False, False]
    assert all(req2.dllm_blocks[i].status == DllmBlockStatus.IN_CACHE for i in range(4))


def test_first_block_does_not_force_decode_topk_without_prev_block() -> None:
    block = DllmBlock(
        block_id=0,
        start=0,
        end=32,
        block_size=32,
        mask_token_id=-1,
        thresholds=SimpleNamespace(
            add_block_threshold=0.1,
            semi_complete_threshold=0.9,
            decoding_threshold=0.9,
        ),
        prev_block=None,
    )

    assert block.should_force_decode_topk is False


def test_request_last_block_finished_is_false_without_prev_block() -> None:
    config = SimpleNamespace(
        block_size=32,
        buffer_size=1,
        mask_token_id=-1,
        decoding_thresholds=SimpleNamespace(
            add_block_threshold=0.1,
            semi_complete_threshold=0.9,
            decoding_threshold=0.9,
        ),
        eos=-1,
        max_model_len=2048,
    )
    req = MultiBDReq(list(range(16)))
    req.page_size = 32
    req.init_multi_block(config)
    req.make_pending()
    req.step()

    assert req.last_block_finished is False
