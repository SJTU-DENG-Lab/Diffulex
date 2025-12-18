import itertools

def build_configs():
    BLOCK_M_LIST = [64, 128, 256]
    BLOCK_N_LIST = [64, 128, 256]
    NUM_STAGES_LIST = [0, 1, 2]
    NUM_THREADS_LIST = [128, 256]
    CONFIGS = list(
        itertools.product(
            BLOCK_M_LIST,
            BLOCK_N_LIST,
            NUM_STAGES_LIST,
            NUM_THREADS_LIST,
        )
    )

    return [
        {
            "BLOCK_M": c[0],
            "BLOCK_N": c[1],
            "NUM_STAGES": c[2],
            "NUM_THREADS": c[3],
        } for c in CONFIGS
    ]