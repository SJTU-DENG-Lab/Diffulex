from dataclasses import dataclass

from diffulex.engine.request import DllmReq
from diffulex.logger import get_logger

logger = get_logger(__name__)


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
    eos_token_generated: bool
    
    text: str = None
    
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
                eos_token_generated=False
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
            cur_trajectory.trajectory.append(
                ReqStep(
                    step_id=len(cur_trajectory.trajectory) - 1,
                    step_time=step_time,
                    is_prefill=req.is_prefilling,
                    num_generated_tokens=req.new_tokens,
                    running_token_ids=req.running_sequence.copy() if req.running_sequence else None,
                    block_size=req.block_size,
                    buffer_bids=[block.block_id for block in req.dllm_block_buffer.dllm_blocks],
                )
            )
            cur_trajectory.token_ids = req.truncated_response.copy() if req.truncated_response else []
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
        
    def convert_to_text(self, tokenizer) :
        for trajectory in self.trajectories:
            trajectory.text = tokenizer.decode(trajectory.token_ids).split(tokenizer.eos_token)[0]

    def to_benchmark_format(self) -> list[dict]:
        """Convert to list of dicts expected by diffulex_bench: text, token_ids, n_diff_steps."""
        return [
            dict(
                text=t.text or "",
                token_ids=t.token_ids if t.token_ids is not None else [],
                n_diff_steps=len(t.trajectory),
            )
            for t in self.trajectories
        ]