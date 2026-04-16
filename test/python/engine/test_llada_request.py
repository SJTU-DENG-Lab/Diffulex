from types import SimpleNamespace

from diffulex.strategy_template.multi_block.engine.request import MultiBlockReqTemplate


def test_multiblock_request_eos_detection_honors_ignore_eos():
    req = SimpleNamespace(
        ignore_eos=True,
        prefix_len=2,
        token_ids=[11, 12, 42, 7],
        eos_token_id=42,
    )

    assert MultiBlockReqTemplate.eos_token_generated.fget(req) is False


def test_multiblock_request_eos_detection_checks_generated_suffix_only():
    req = SimpleNamespace(
        ignore_eos=False,
        prefix_len=2,
        token_ids=[42, 12, 7, 8],
        eos_token_id=42,
    )

    assert MultiBlockReqTemplate.eos_token_generated.fget(req) is False
