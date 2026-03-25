from dataclasses import dataclass, field

from diffulex.engine.request import DllmReq
from diffulex.logger import get_logger

logger = get_logger(__name__)


def decode_token_ids_robust(tokenizer, token_ids: list[int] | None, *, skip_special_tokens: bool = False) -> str:
    """Decode token ids to text.

    Some checkpoints emit ids that ``convert_ids_to_tokens`` maps to ``None`` (tokenizer/model vocab skew).
    Qwen2/GPT2 ``convert_tokens_to_string`` then does ``"".join(tokens)`` and crashes with ``TypeError``.
    """
    if not token_ids:
        return ""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    except TypeError:
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        safe = [t if t is not None else "" for t in tokens]
        return tokenizer.convert_tokens_to_string(safe)


@dataclass
class ReqStep:
    step_id: int
    step_time: float

    is_prefill: bool

    num_generated_tokens: int
    running_token_ids: list[int]

    block_size: int
    buffer_bids: list[int]

    # Populated when engine saves KV mapping trace (multi-block prepare); see Config.save_kv_mapping_trace.
    kv_mapping_trace: dict | None = None
    # Populated from sampler output maps during scheduler postprocess for block-by-block diffing.
    sampler_trace: dict | None = None

    def to_dict(self) -> dict:
        d = dict(
            step_id=self.step_id,
            step_time=self.step_time,
            is_prefill=self.is_prefill,
            num_generated_tokens=self.num_generated_tokens,
            running_token_ids=self.running_token_ids,
            block_size=self.block_size,
            buffer_bids=self.buffer_bids,
        )
        if self.kv_mapping_trace is not None:
            d["kv_mapping_trace"] = self.kv_mapping_trace
        if self.sampler_trace is not None:
            d["sampler_trace"] = self.sampler_trace
        return d


@dataclass
class ReqTrajectory:
    req_id: int

    token_ids: list[int]
    trajectory: list[ReqStep]

    is_truncated: bool
    max_new_tokens_reached: bool
    max_model_len_reached: bool
    eos_token_generated: bool

    text: str = None
    # Generation-only tokens including content after EOS (when applicable); see DllmReq.full_response.
    full_token_ids: list[int] = field(default_factory=list)
    full_text: str | None = None

    def to_dict(self) -> dict:
        return dict(
            req_id=self.req_id,
            token_ids=self.token_ids,
            trajectory=[step.to_dict() for step in self.trajectory],
            is_truncated=self.is_truncated,
            max_new_tokens_reached=self.max_new_tokens_reached,
            max_model_len_reached=self.max_model_len_reached,
            eos_token_generated=self.eos_token_generated,
            text=self.text,
        )


class GenerationOutputs:
    """Accumulates generation outputs."""

    def __init__(self, num_prompts: int):
        self.trajectories: list[ReqTrajectory] = [
            ReqTrajectory(
                req_id=req_id,
                token_ids=[],
                trajectory=[],
                is_truncated=False,
                max_new_tokens_reached=False,
                max_model_len_reached=False,
                eos_token_generated=False,
            )
            for req_id in range(num_prompts)
        ]

    @property
    def tpf(self) -> float:
        num_generated_tokens = 0
        num_steps = 0
        for trajectory in self.trajectories:
            for step in trajectory.trajectory:
                num_generated_tokens += step.num_generated_tokens
                num_steps += 1
        return num_generated_tokens / num_steps if num_steps > 0 else 0

    @property
    def ttft(self) -> float:
        num_generated_tokens = 0
        total_time = 0.0
        for trajectory in self.trajectories:
            if len(trajectory.trajectory) == 0:
                continue

            prefill_step = trajectory.trajectory[0]
            if not prefill_step.is_prefill:
                continue

            num_generated_tokens += prefill_step.num_generated_tokens
            total_time += prefill_step.step_time
        return total_time / num_generated_tokens if num_generated_tokens > 0 else 0

    @property
    def tpot(self) -> float:
        return 1 / self.decode_throughput if self.decode_throughput > 0 else 0

    @property
    def prefill_throughput(self) -> float:
        num_prefill_tokens = 0
        total_time = 0.0
        for trajectory in self.trajectories:
            if len(trajectory.trajectory) == 0:
                continue

            prefill_step = trajectory.trajectory[0]
            if not prefill_step.is_prefill:
                continue

            num_prefill_tokens += len(prefill_step.running_token_ids)
            total_time += prefill_step.step_time
        return num_prefill_tokens / total_time

    @property
    def decode_throughput(self) -> float:
        num_generated_tokens = 0
        total_time = 0.0
        for trajectory in self.trajectories:
            for step in trajectory.trajectory:
                if not step.is_prefill:
                    num_generated_tokens += step.num_generated_tokens
                    total_time += step.step_time
        return num_generated_tokens / total_time if total_time > 0 else 0

    def record_step(self, reqs: list[DllmReq], step_time: float, req_id_to_prompt_id: dict[int, int] | None = None):
        for req in reqs:
            prompt_idx = (req_id_to_prompt_id or {}).get(req.req_id, req.req_id)
            if prompt_idx >= len(self.trajectories):
                continue
            cur_trajectory = self.trajectories[prompt_idx]
            kv_trace = getattr(req, "last_kv_mapping_trace", None)
            sampler_trace = getattr(req, "last_sampler_trace", None)
            step_id = len(cur_trajectory.trajectory)
            cur_trajectory.trajectory.append(
                ReqStep(
                    step_id=step_id,
                    step_time=step_time,
                    is_prefill=req.is_prefilling,
                    num_generated_tokens=req.new_tokens,
                    running_token_ids=req.running_sequence.copy() if req.running_sequence else None,
                    block_size=req.block_size,
                    buffer_bids=[block.block_id for block in req.dllm_block_buffer.dllm_blocks],
                    kv_mapping_trace=kv_trace,
                    sampler_trace=sampler_trace,
                )
            )
            cur_trajectory.token_ids = req.truncated_response.copy() if req.truncated_response else []
            if hasattr(req, "full_response"):
                try:
                    cur_trajectory.full_token_ids = list(req.full_response)
                except Exception:
                    pass
            cur_trajectory.is_truncated = req.is_truncated
            cur_trajectory.max_new_tokens_reached = req.max_new_tokens_reached
            cur_trajectory.max_model_len_reached = req.max_model_len_reached
            cur_trajectory.eos_token_generated = req.eos_token_generated

    def postfix(self) -> dict:
        return dict(
            tpf=f"{int(self.tpf)}",
            ttft=f"{self.ttft:.2f}",
            tpot=f"{self.tpot:.2f}",
            ptps=f"{int(self.prefill_throughput)}tok/sec",
            dtps=f"{int(self.decode_throughput)}tok/sec",
        )

    def log_summary(self):
        logger.info("--------------------------------")
        logger.info("Generation Outputs Summary:")
        logger.info("--------------------------------")
        logger.info(f"Total Tokens: {sum(len(trajectory.token_ids) for trajectory in self.trajectories)} toks")
        logger.info(f"Total NFEs: {sum(len(trajectory.trajectory) for trajectory in self.trajectories)} nfes (steps)")
        logger.info(f"Total Time: {sum(trajectory.trajectory[-1].step_time for trajectory in self.trajectories)} sec")
        logger.info(f"TPF: {self.tpf:.2f} tok/step")
        logger.info(f"TTFT: {self.ttft:.2f} tok/sec")
        logger.info(f"TPOT: {self.tpot:.2f} tok/sec")
        logger.info(f"Prefill Throughput: {self.prefill_throughput:.2f} tok/sec")
        logger.info(f"Decode Throughput: {self.decode_throughput:.2f} tok/sec")
        logger.info("--------------------------------")

    def convert_to_text(self, tokenizer):
        eos = getattr(tokenizer, "eos_token", None) or ""
        for trajectory in self.trajectories:
            gen_full = trajectory.full_token_ids if trajectory.full_token_ids else trajectory.token_ids
            raw_full = decode_token_ids_robust(tokenizer, gen_full)
            trajectory.full_text = raw_full
            raw_trunc = decode_token_ids_robust(tokenizer, trajectory.token_ids)
            trajectory.text = raw_trunc.split(eos)[0] if eos else raw_trunc

    def to_benchmark_format(self) -> list[dict]:
        """Convert to list of dicts expected by diffulex_bench: text, token_ids, n_diff_steps."""
        return [
            dict(
                text=t.text or "",
                full_text=(t.full_text if t.full_text is not None else t.text or ""),
                token_ids=t.token_ids if t.token_ids is not None else [],
                n_diff_steps=len(t.trajectory),
            )
            for t in self.trajectories
        ]
