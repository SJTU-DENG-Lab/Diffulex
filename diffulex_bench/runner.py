"""
Benchmark Runner - Benchmark runner that wraps Diffulex inference engine
Provides a unified interface for benchmarking
"""

import time
from typing import List, Dict, Any, Optional

from diffulex import Diffulex, SamplingParams
from transformers import AutoTokenizer
from diffulex.logger import get_logger


class BenchmarkRunner:
    """
    Benchmark runner that wraps the Diffulex inference engine
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        wait_ready: bool = True,
        **diffulex_kwargs,
    ):
        """
        Initialize the benchmark runner

        Args:
            model_path: Path to the model
            tokenizer_path: Path to the tokenizer, if None uses model_path
            wait_ready: Whether to wait for engine to be fully initialized before returning
            **diffulex_kwargs: Additional arguments to pass to Diffulex
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.logger = get_logger(__name__)

        # Initialize Diffulex engine
        self.logger.info("Initializing Diffulex engine...")
        self.llm = Diffulex(model_path, **diffulex_kwargs)

        # Wait for engine to be ready if requested
        if wait_ready:
            self._wait_for_ready()

        # Load tokenizer
        self.logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.logger.success("Tokenizer loaded successfully")

    def _wait_for_ready(self, timeout: float = 300.0, check_interval: float = 0.5):
        """
        Wait for the Diffulex engine to be fully initialized and ready

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Interval between readiness checks in seconds
        """
        start_time = time.time()

        if hasattr(self.llm, "ps") and self.llm.ps:
            num_subprocesses = len(self.llm.ps)
            self.logger.info(f"Waiting for {num_subprocesses} engine subprocess(es) to be ready...")

            while time.time() - start_time < timeout:
                all_alive = all(p.is_alive() for p in self.llm.ps)

                if all_alive:
                    time.sleep(2.0)
                    self.logger.success("All engine subprocesses are ready")
                    return

                dead_processes = [i for i, p in enumerate(self.llm.ps) if not p.is_alive()]
                exit_codes = [self.llm.ps[i].exitcode for i in dead_processes]
                raise RuntimeError(
                    f"Engine subprocess(es) {dead_processes} terminated during initialization. "
                    f"Exit code(s): {exit_codes}"
                )

            elapsed = time.time() - start_time
            raise RuntimeError(f"Timeout waiting for engine subprocesses to be ready after {elapsed:.1f}s")

        self.logger.success("Engine is ready")
        return

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate text

        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters
            use_tqdm: Whether to show progress bar
        Returns:
            List of generation results, each containing text, token_ids, nfe
        """
        start_time = time.time()

        raw_outputs = self.llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
        end_time = time.time()

        # Convert GenerationOutputs to list of dicts if needed (tp_worker returns GenerationOutputs)
        batch_metrics = {}
        if hasattr(raw_outputs, "to_benchmark_format"):
            outputs = raw_outputs.to_benchmark_format()
            batch_metrics = {
                "ttft_s": getattr(raw_outputs, "ttft", 0.0),
                "tpot_s": getattr(raw_outputs, "tpot", 0.0),
                "e2e_total_time_s": getattr(raw_outputs, "e2e_total_time", 0.0),
                "e2e_throughput_tok_s": getattr(raw_outputs, "e2e_throughput", 0.0),
                "prefill_throughput_tok_s": getattr(raw_outputs, "prefill_throughput", 0.0),
                "decode_throughput_tok_s": getattr(raw_outputs, "decode_throughput", 0.0),
                "batch_total_time_s": getattr(raw_outputs, "total_time", 0.0),
                "tpf": getattr(raw_outputs, "tpf", 0.0),
            }
        else:
            outputs = raw_outputs

        # Add timing information
        total_time = end_time - start_time
        for output in outputs:
            output["generation_time"] = total_time / len(outputs) if outputs else 0
            output.update(batch_metrics)

        return outputs

    def evaluate_batch(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of prompts

        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters
            use_tqdm: Whether to show progress bar

        Returns:
            Evaluation result dictionary containing generation results and statistics
        """
        outputs = self.generate(prompts, sampling_params, use_tqdm=use_tqdm)

        # Calculate statistics
        total_tokens = sum(len(o["token_ids"]) for o in outputs)
        total_time = sum(o.get("generation_time", 0) for o in outputs)
        avg_nfe = sum(o.get("nfe", o.get("num_nfes", o.get("n_diff_steps", 0))) for o in outputs) / len(outputs) if outputs else 0
        total_nfe = sum(o.get("nfe", o.get("num_nfes", o.get("n_diff_steps", 0))) for o in outputs)

        return {
            "outputs": outputs,
            "num_samples": len(outputs),
            "total_tokens": total_tokens,
            "total_nfe": total_nfe,
            "total_time": total_time,
            "avg_tokens_per_sample": total_tokens / len(outputs) if outputs else 0,
            "avg_nfe": avg_nfe,
            "tpf": total_tokens / total_nfe if total_nfe > 0 else 0,
            "e2e_total_time_s": outputs[0].get("e2e_total_time_s", 0.0) if outputs else 0.0,
            "ttft_s": outputs[0].get("ttft_s", 0.0) if outputs else 0.0,
            "tpot_s": outputs[0].get("tpot_s", 0.0) if outputs else 0.0,
            "e2e_throughput_tok_s": outputs[0].get("e2e_throughput_tok_s", 0.0) if outputs else 0.0,
            "prefill_throughput_tok_s": outputs[0].get("prefill_throughput_tok_s", 0.0) if outputs else 0.0,
            "decode_throughput_tok_s": outputs[0].get("decode_throughput_tok_s", 0.0) if outputs else 0.0,
        }
