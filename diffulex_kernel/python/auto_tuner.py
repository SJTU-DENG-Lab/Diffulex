import itertools



def build_chunked_prefill_configs():
    """Autotune configs for chunked prefill unified kernel.

    Note: The autotune key is
    [
        "NUM_GROUPS",
        "HEAD_DIM",
        "PAGE_SIZE",
        "DLLM_BLOCK_SIZE",
        "IS_BLOCK_CAUSAL",
        "IS_PREFIX_FULL",
        "MASK_PREFIX_HOLE",
        "PREFIX_CAUSAL",
    ],
    so configs are selected based on those runtime parameters.
    """
    BLOCK_M_LIST = [64, 128]
    BLOCK_N_LIST = [64, 128]
    NUM_STAGES_LIST = [1, 2, 3]
    NUM_WARPS_LIST = [4, 8]

    CONFIGS = list(
        itertools.product(
            BLOCK_M_LIST,
            BLOCK_N_LIST,
            NUM_STAGES_LIST,
            NUM_WARPS_LIST,
        )
    )

    return [
        {
            "BLOCK_M": c[0],
            "BLOCK_N": c[1],
            "num_stages": c[2],
            "num_warps": c[3],
        }
        for c in CONFIGS
    ]
