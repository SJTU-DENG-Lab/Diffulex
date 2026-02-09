import torch
import tilelang
import tilelang.language as T

from flash_attn import flash_attn_varlen_func
from tilelang.autotuner import set_autotune_inputs

from diffulex_kernel.python.auto_tuner import build_configs
from diffulex.attention.metadata import AttnMetaDataBase, is_warming_up


kernel_config = None


@tilelang.autotune(configs=build_configs())
@tilelang.jit(
    # NOTE: Disable TMA and warp specialized for now to avoid compile error on Hopper
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def dllm_flash_attn_prefill_kernel(
    NUM_SEQS: int,
    NUM_GROUPS: int,
    Q_LEN: int,
    KV_LEN: int,
    NUM_HEADS: int,
    HEAD_DIM: int,
    IS_BLOCK_ATTN: bool,
    DIFFUSION_BLOCK_SIZE: int,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    NUM_STAGES: int = 1,
    NUM_THREADS: int = 128,
):
    SCALE = (1.0 / HEAD_DIM) ** 0.5 * 1.44269504  # log2(e)
    NUM_KV_HEADS = NUM_HEADS // NUM_GROUPS
    Q_SHAPE = [Q_LEN, NUM_HEADS, HEAD_DIM]
    KV_SHAPE = [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
    O_SHAPE = [Q_LEN, NUM_HEADS, HEAD_DIM]
    DTYPE = "bfloat16"
    ACCUM_DTYPE = "float"

    @T.prim_func
    def kernel(
        Q: T.Tensor(Q_SHAPE, DTYPE),
        K: T.Tensor(KV_SHAPE, DTYPE),
        V: T.Tensor(KV_SHAPE, DTYPE),
        cu_seqlens_q: T.Tensor(NUM_SEQS + 1, "int32"),
        cu_seqlens_k: T.Tensor(NUM_SEQS + 1, "int32"),
        max_seqlen_q: T.int32,
        O: T.Tensor(O_SHAPE, DTYPE),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, BLOCK_M), NUM_HEADS, NUM_SEQS, threads=NUM_THREADS) as (bx, by, bz):
            Q_shared = T.alloc_shared([BLOCK_M, HEAD_DIM], DTYPE)
            K_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], DTYPE)
            V_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], DTYPE)
            O_shared = T.alloc_shared([BLOCK_M, HEAD_DIM], DTYPE)

            acc_score = T.alloc_fragment([BLOCK_M, BLOCK_N], ACCUM_DTYPE)
            acc_score_cast = T.alloc_fragment([BLOCK_M, BLOCK_N], DTYPE)
            acc_output = T.alloc_fragment([BLOCK_M, HEAD_DIM], ACCUM_DTYPE)
            scores_max = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_max_prev = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_scale = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_sum = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            log_sum = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)

            T.annotate_layout(
                {
                    Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                    O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                }
            )

            q_block_idx = bx
            seq_idx = bz
            head_idx = by
            kv_head_idx = head_idx // NUM_GROUPS

            q_start_idx = cu_seqlens_q[seq_idx]
            kv_start_idx = cu_seqlens_k[seq_idx]
            q_end_idx = cu_seqlens_q[seq_idx + 1]
            kv_end_idx = cu_seqlens_k[seq_idx + 1]

            cur_q_seqlen = q_end_idx - q_start_idx
            cur_kv_seqlen = kv_end_idx - kv_start_idx

            T.copy(
                Q[q_start_idx + q_block_idx * BLOCK_M : q_start_idx + (q_block_idx + 1) * BLOCK_M, head_idx, :],
                Q_shared,
            )

            T.fill(acc_output, 0)
            T.fill(acc_score, 0)
            T.fill(log_sum, 0)
            T.fill(scores_max, -T.infinity(ACCUM_DTYPE))

            loop_range = (
                T.min(
                    T.ceildiv(cur_q_seqlen + (q_block_idx + 1) * BLOCK_M, BLOCK_N),
                    T.ceildiv(cur_kv_seqlen, BLOCK_N),
                )
                if IS_BLOCK_ATTN
                else T.ceildiv(cur_kv_seqlen, BLOCK_N)
            )
            for kv_block_idx in T.Pipelined(loop_range, num_stages=NUM_STAGES):
                T.copy(
                    K[kv_start_idx + kv_block_idx * BLOCK_N : kv_start_idx + (kv_block_idx + 1) * BLOCK_N, kv_head_idx, :],
                    K_shared,
                )

                if IS_BLOCK_ATTN:
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                        num_diffusion_blocks = (q_block_idx * BLOCK_M + i) // DIFFUSION_BLOCK_SIZE + 1
                        acc_score[i, j] = T.if_then_else(
                            (num_diffusion_blocks * DIFFUSION_BLOCK_SIZE <= kv_block_idx * BLOCK_N + j)
                            or (q_block_idx * BLOCK_M + i >= cur_q_seqlen or kv_block_idx * BLOCK_N + j >= cur_kv_seqlen),
                            -1e9,
                            0,
                        )
                else:
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                        acc_score[i, j] = T.if_then_else(
                            (q_block_idx * BLOCK_M + i >= cur_q_seqlen or kv_block_idx * BLOCK_N + j >= cur_kv_seqlen),
                            -1e9,
                            0,
                        )

                T.gemm(Q_shared, K_shared, acc_score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(ACCUM_DTYPE))
                T.reduce_max(acc_score, scores_max, dim=1, clear=False)
                for i in T.Parallel(BLOCK_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.parallel(BLOCK_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * SCALE - scores_max[i] * SCALE)

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    acc_score[i, j] = T.exp2(acc_score[i, j] * SCALE - scores_max[i] * SCALE)

                T.reduce_sum(acc_score, scores_sum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    log_sum[i] = log_sum[i] * scores_scale[i] + scores_sum[i]

                T.copy(acc_score, acc_score_cast)
                for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
                    acc_output[i, j] *= scores_scale[i]

                T.copy(
                    V[kv_start_idx + kv_block_idx * BLOCK_N : kv_start_idx + (kv_block_idx + 1) * BLOCK_N, kv_head_idx, :],
                    V_shared,
                )
                T.gemm(acc_score_cast, V_shared, acc_output, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
                acc_output[i, j] /= log_sum[i]

            T.copy(acc_output, O_shared)
            for i, d_idx in T.Parallel(BLOCK_M, HEAD_DIM):
                if i + q_block_idx * BLOCK_M < cur_q_seqlen:
                    O[i + q_start_idx + q_block_idx * BLOCK_M, head_idx, d_idx] = O_shared[i, d_idx]

    return kernel


def dllm_flash_attn_prefill_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_metadata: AttnMetaDataBase,
) -> torch.Tensor:
    """
    TileLang-based prefill implementation (existing behavior).
    Kept in a separate module so importing decode kernels doesn't require TileLang.
    """
    global kernel_config
    if attn_metadata.attn_type == "full_attention":
        return flash_attn_varlen_func(
            q,
            k,
            v,
            attn_metadata.cu_seqlens_q,
            attn_metadata.cu_seqlens_k,
            attn_metadata.max_seqlen_q,
            attn_metadata.max_seqlen_k,
            softmax_scale=scale,
            block_table=None,
        )
    if attn_metadata.attn_type != "block_attention":
        raise ValueError(f"Unsupported attn_type={attn_metadata.attn_type!r} for prefill")

    if is_warming_up():
        with set_autotune_inputs(
            [
                q,
                k,
                v,
                attn_metadata.cu_seqlens_q,
                attn_metadata.cu_seqlens_k,
                attn_metadata.max_seqlen_q,
            ]
        ):
            prefill_kernel = dllm_flash_attn_prefill_kernel(
                attn_metadata.num_seqs,
                q.shape[1] // k.shape[1],
                q.shape[0],
                k.shape[0],
                q.shape[1],
                q.shape[2],
                attn_metadata.attn_type == "block_attention",
                attn_metadata.diffusion_block_size,
            )
        kernel_config = prefill_kernel.config
        return prefill_kernel(
            q,
            k,
            v,
            attn_metadata.cu_seqlens_q,
            attn_metadata.cu_seqlens_k,
            attn_metadata.max_seqlen_q,
        )

    config_kwargs = kernel_config if kernel_config is not None else {}
    prefill_kernel = dllm_flash_attn_prefill_kernel(
        attn_metadata.num_seqs,
        q.shape[1] // k.shape[1],
        q.shape[0],
        k.shape[0],
        q.shape[1],
        q.shape[2],
        attn_metadata.attn_type == "block_attention",
        attn_metadata.diffusion_block_size,
        **config_kwargs,
    )
    return prefill_kernel(
        q,
        k,
        v,
        attn_metadata.cu_seqlens_q,
        attn_metadata.cu_seqlens_k,
        attn_metadata.max_seqlen_q,
    )

