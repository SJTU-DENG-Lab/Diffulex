from dataclasses import dataclass, field

from diffulex.engine.request import DllmReq
from diffulex.logger import get_logger

logger = get_logger(__name__)


def decode_token_ids_robust(tokenizer, token_ids: list[int] | None, *, skip_special_tokens: bool = False) -> str:
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

    def to_dict(self) -> dict:
        return dict(
            step_id=self.step_id,
            step_time=self.step_time,
            is_prefill=self.is_prefill,
            num_generated_tokens=self.num_generated_tokens,
            running_token_ids=self.running_token_ids,
            block_size=self.block_size,
            buffer_bids=self.buffer_bids,
        )


@dataclass
class ReqTrajectory:
    req_id: int

    token_ids: list[int]
    trajectory: list[ReqStep]

    is_truncated: bool
    max_new_tokens_reached: bool
    max_model_len_reached: bool
    max_nfe_reached: bool
    max_repetition_run_reached: bool
    eos_token_generated: bool
    completion_reason: str | None = None

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
            max_nfe_reached=self.max_nfe_reached,
            max_repetition_run_reached=self.max_repetition_run_reached,
            eos_token_generated=self.eos_token_generated,
            completion_reason=self.completion_reason,
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
                max_nfe_reached=False,
                max_repetition_run_reached=False,
                eos_token_generated=False,
                completion_reason=None,
            )
            for req_id in range(num_prompts)
        ]
        self._batch_step_count = 0
        self._batch_total_time = 0.0
        self._batch_generated_tokens = 0
        self._prefill_batch_time = 0.0
        self._prefill_batch_tokens = 0
        self._decode_batch_time = 0.0
        self._decode_batch_tokens = 0

    @property
    def batch_step_count(self) -> int:
        return self._batch_step_count

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    
    @property
    def tpf(self) -> float:
        per_req_tpf = []
        for trajectory in self.trajectories:
            if not trajectory.trajectory:
                continue
            num_generated_tokens = sum(step.num_generated_tokens for step in trajectory.trajectory)
            per_req_tpf.append(num_generated_tokens / len(trajectory.trajectory))
        return self._mean(per_req_tpf)

    @property
    def ttft(self) -> float:
        per_req_ttft = []
        for trajectory in self.trajectories:
            elapsed = 0.0
            for step in trajectory.trajectory:
                elapsed += step.step_time
                if step.num_generated_tokens > 0:
                    per_req_ttft.append(elapsed)
                    break
        return self._mean(per_req_ttft)

    @property
    def tpot(self) -> float:
        per_req_tpot = []
        for trajectory in self.trajectories:
            total_time = sum(step.step_time for step in trajectory.trajectory)
            total_generated_tokens = sum(step.num_generated_tokens for step in trajectory.trajectory)
            if total_generated_tokens <= 1:
                continue

            elapsed = 0.0
            ttft = None
            for step in trajectory.trajectory:
                elapsed += step.step_time
                if step.num_generated_tokens > 0:
                    ttft = elapsed
                    break
            if ttft is not None:
                per_req_tpot.append((total_time - ttft) / (total_generated_tokens - 1))
        return self._mean(per_req_tpot)

    @property
    def throughput(self) -> float:
        return self._batch_generated_tokens / self._batch_total_time if self._batch_total_time > 0 else 0

    @property
    def prefill_throughput(self) -> float:
        return self._prefill_batch_tokens / self._prefill_batch_time if self._prefill_batch_time > 0 else 0

    @property
    def decode_throughput(self) -> float:
        return self._decode_batch_tokens / self._decode_batch_time if self._decode_batch_time > 0 else 0

    @property
    def total_time(self) -> float:
        return self._batch_total_time

    def record_step(self, reqs: list[DllmReq], step_time: float, req_id_to_prompt_id: dict[int, int] | None = None):
        if reqs:
            self._batch_step_count += 1
            self._batch_total_time += step_time

            has_prefill = False
            has_decode = False
            prefill_tokens_this_step = 0
            decode_tokens_this_step = 0
            generated_tokens_this_step = 0

            for req in reqs:
                generated_tokens_this_step += req.new_tokens
                running_sequence = req.running_sequence
                if req.is_prefilling:
                    has_prefill = True
                    prefill_tokens_this_step += len(running_sequence or [])
                else:
                    has_decode = True
                    decode_tokens_this_step += req.new_tokens

            self._batch_generated_tokens += generated_tokens_this_step
            self._prefill_batch_tokens += prefill_tokens_this_step
            self._decode_batch_tokens += decode_tokens_this_step
            if has_prefill:
                self._prefill_batch_time += step_time
            if has_decode:
                self._decode_batch_time += step_time

        for req in reqs:
            prompt_idx = (req_id_to_prompt_id or {}).get(req.req_id, req.req_id)
            if prompt_idx >= len(self.trajectories):
                continue
            cur_trajectory = self.trajectories[prompt_idx]
            step_id = len(cur_trajectory.trajectory)
            cur_trajectory.trajectory.append(
                ReqStep(
                    step_id=step_id,
                    step_time=step_time,
                    is_prefill=req.is_prefilling,
                    num_generated_tokens=req.new_tokens,
                    running_token_ids=(
                        req.running_sequence.copy() if req.running_sequence is not None else []
                    ),
                    block_size=req.block_size,
                    buffer_bids=[block.block_id for block in req.dllm_block_buffer.dllm_blocks],
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
            cur_trajectory.max_nfe_reached = getattr(req, "max_nfe_reached", False)
            cur_trajectory.max_repetition_run_reached = getattr(req, "max_repetition_run_reached", False)
            cur_trajectory.eos_token_generated = req.eos_token_generated
            cur_trajectory.completion_reason = getattr(req, "completion_reason", None)

    def postfix(self) -> dict:
        return dict(
            tpf=f"{self.tpf:.2f}tok/step",
            ttft=f"{self.ttft:.2f}s",
            tpot=f"{self.tpot:.2f}s",
            ptps=f"{self.prefill_throughput:.2f}tok/s",
            dtps=f"{self.decode_throughput:.2f}tok/s",
            tps=f"{self.throughput:.2f}tok/s",
        )

    def log_summary(self):
        logger.info("--------------------------------")
        logger.info("Generation Outputs Summary:")
        logger.info("--------------------------------")
        logger.info(f"Total Tokens: {sum(len(trajectory.token_ids) for trajectory in self.trajectories)} toks")
        logger.info(f"Total NFEs: {self.batch_step_count} nfes (steps)")
        logger.info(f"Total Time: {self.total_time} sec")
        logger.info(f"TPF: {self.tpf:.2f} tok/step")
        logger.info(f"TTFT: {self.ttft:.2f} sec")
        logger.info(f"TPOT: {self.tpot:.2f} sec")
        logger.info(f"Throughput: {self.throughput:.2f} tok/sec")
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
        """Convert to list of dicts expected by diffulex_bench: text, token_ids, nfe."""
        return [
            dict(
                text=t.text or "",
                full_text=(t.full_text if t.full_text is not None else t.text or ""),
                token_ids=t.token_ids if t.token_ids is not None else [],
                nfe=len(t.trajectory),
            )
            for t in self.trajectories
        ]
