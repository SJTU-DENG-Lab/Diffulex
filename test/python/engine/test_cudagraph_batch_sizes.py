from diffulex.strategy_template.multi_block.engine.model_runner import MultiBlockModelRunnerTemplate


def test_graph_seq_batch_sizes_stay_within_capacity_and_include_tail() -> None:
    seq_bs = MultiBlockModelRunnerTemplate._graph_seq_batch_sizes(24)

    assert seq_bs == [1, 2, 4, 8, 16, 24]
    assert max(seq_bs) == 24


def test_graph_seq_batch_sizes_handles_small_limits() -> None:
    assert MultiBlockModelRunnerTemplate._graph_seq_batch_sizes(3) == [1, 2, 3]
