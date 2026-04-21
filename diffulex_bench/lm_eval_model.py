"""
LM Eval Model - Diffulex integration with lm-evaluation-harness
"""

import logging
import os
import re
import time
import json
from typing import List, Optional, Tuple, Type, TypeVar, Union

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from diffulex import SamplingParams
from diffulex.utils.output import decode_token_ids_robust
from diffulex_bench.runner import BenchmarkRunner
from diffulex_bench.config import decode_model_arg_value, extract_diffulex_engine_kwargs
from diffulex.logger import get_logger

T = TypeVar("T", bound="LM")
eval_logger = logging.getLogger(__name__)


def _compact_numeric_arrays_in_json(json_str: str) -> str:
    """Collapse whitespace inside numeric JSON arrays (same idea as multi_bd/eval/main.py)."""
    return re.sub(
        r"\[\s*([\d\.\,\-\+eE\s]+?)\s*\]",
        lambda m: "[" + m.group(1).replace("\n", "").replace(" ", "") + "]",
        json_str,
    )


def _normalize_until_terms(until: object) -> list[str]:
    if until is None:
        return []
    if isinstance(until, str):
        return [until] if until else []
    if isinstance(until, (list, tuple)):
        return [str(x) for x in until if x is not None and str(x) != ""]
    return []


def _strip_at_until_terms(response: str, until_terms: list[str]) -> str:
    """Align with multi_bd ``postprocess_generate_until`` when escape_until is False."""
    out = response
    for term in until_terms:
        if term:
            out = out.split(term)[0]
    return out


def _coerce_bool(v: Union[bool, str, int, None], default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


@register_model("diffulex")
class DiffulexLM(LM):
    """
    Diffulex model integration for lm-evaluation-harness
    """

    def __init__(
        self,
        pretrained: str,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, type]] = "auto",
        max_new_tokens: Optional[int] = 256,
        max_nfe: Optional[int] = None,
        max_repetition_run: Optional[int] = None,
        max_length: Optional[int] = 2048,
        add_bos_token: Optional[bool] = False,
        trust_remote_code: Optional[bool] = True,
        temperature: Optional[float] = 0.0,
        model_name: Optional[str] = "dream",
        decoding_strategy: Optional[str] = "d2f",
        sampling_mode: Optional[str] = "naive",
        mask_token_id: Optional[int] = 151666,
        tensor_parallel_size: Optional[int] = 1,
        data_parallel_size: Optional[int] = 1,
        expert_parallel_size: Optional[int] = 1,
        gpu_memory_utilization: Optional[float] = 0.9,
        max_model_len: Optional[int] = 2048,
        max_num_batched_tokens: Optional[int] = 4096,
        max_num_reqs: Optional[int] = 128,
        use_lora: Optional[bool] = False,
        lora_path: Optional[str] = "",
        pre_merge_lora: Optional[bool] = True,
        enforce_eager: Optional[bool] = False,
        enable_prefix_caching: Optional[bool] = True,
        kv_cache_layout: Optional[str] = "unified",
        page_size: Optional[int] = 32,
        token_merge_mode: Optional[str] = "dmax_topk",
        token_merge_top_k: Optional[int] = 1,
        token_merge_renormalize: Optional[bool] = True,
        token_merge_weight: Optional[float] = 1.0,
        decoding_thresholds: Optional[dict] = None,
        add_block_threshold: Optional[float] = None,
        semi_complete_threshold: Optional[float] = None,
        accept_threshold: Optional[float] = None,
        remask_threshold: Optional[float] = None,
        block_size: Optional[int] = 32,
        buffer_size: Optional[int] = 4,
        multi_block_prefix_full: Optional[bool] = False,
        save_dir: Optional[str] = None,
        wait_ready: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # Setup logger
        self.logger = get_logger(__name__)

        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        self.pretrained = pretrained
        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)

        self.max_length = max_length
        self.add_bos_token = add_bos_token
        self.max_new_tokens = max_new_tokens
        self.max_nfe = max_nfe
        self.max_repetition_run = max_repetition_run
        self.temperature = temperature
        self.save_dir = save_dir
        # Cumulative per-eval-run, same layout as multi_bd/eval (rank-0 JSON lists).
        self._responses_full: List[str] = []
        self._responses_truncated: List[str] = []
        self._responses_extracted: List[str] = []

        # Diffulex-specific parameters
        self.model_name = model_name
        self.decoding_strategy = decoding_strategy
        self.sampling_mode = sampling_mode
        self.mask_token_id = mask_token_id

        # Statistics tracking
        self.total_generated_tokens = 0
        self.total_nfe = 0  # Number of Forward Evaluations (diffusion steps)
        self.total_generation_time = 0.0
        self.total_samples = 0
        self.all_generation_times = []
        self.all_nfe = []
        self.all_tokens = []
        self.last_ttft = 0.0
        self.last_tpot = 0.0
        self.last_e2e_total_time = 0.0
        self.last_e2e_throughput = 0.0
        self.last_prefill_throughput = 0.0
        self.last_decode_throughput = 0.0

        engine_sources = locals().copy()
        extra_engine_kwargs = engine_sources.pop("kwargs")
        engine_sources.pop("self", None)
        engine_sources.update(extra_engine_kwargs)

        # Initialize Diffulex runner
        self.runner = BenchmarkRunner(
            model_path=pretrained,
            tokenizer_path=pretrained,
            wait_ready=wait_ready,
            **extract_diffulex_engine_kwargs(engine_sources),
        )

        self.tokenizer = self.runner.tokenizer

        # Create sampling params
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            max_nfe=max_nfe,
            max_repetition_run=max_repetition_run,
        )

        self.logger.success("Diffulex engine initialized successfully")

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return "cuda"  # Diffulex manages device internally

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def tok_decode(self, tokens, skip_special_tokens=True):
        """Decode tokens to text"""
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
            return [
                decode_token_ids_robust(self.tokenizer, t, skip_special_tokens=skip_special_tokens)
                for t in tokens
            ]
        return decode_token_ids_robust(
            self.tokenizer, tokens, skip_special_tokens=skip_special_tokens
        )

    def tok_encode(self, text, add_special_tokens=True):
        """Encode text to tokens"""
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids

    @classmethod
    def create_from_arg_string(cls: Type[T], arg_string: str, additional_config: Optional[dict] = None) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.

        Args:
            arg_string: A string containing arguments in the format key1=value1,key2=value2
            additional_config: Optional dictionary containing additional configuration parameters

        Returns:
            Instance of the LM class
        """
        additional_config = {} if additional_config is None else additional_config
        args = {
            k: decode_model_arg_value(v)
            for k, v in utils.simple_parse_args_string(arg_string).items()
        }
        args2 = {
            k: decode_model_arg_value(v)
            for k, v in additional_config.items()
            if v is not None
        }
        return cls(**args, **args2)

    @classmethod
    def create_from_arg_obj(cls: Type[T], arg_dict: dict, additional_config: Optional[dict] = None) -> T:
        """Mirror lm-eval's dict-based init path while decoding encoded complex values."""
        additional_config = {} if additional_config is None else additional_config
        args = {
            k: decode_model_arg_value(v)
            for k, v in arg_dict.items()
        }
        args2 = {
            k: decode_model_arg_value(v)
            for k, v in additional_config.items()
            if v is not None
        }
        return cls(**args, **args2)

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
        """
        Apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        """
        Generate text until stopping conditions are met.

        Args:
            requests: List of generation requests
            disable_tqdm: Whether to disable progress bar

        Returns:
            List of generated texts
        """
        self.logger.info(f"Processing {len(requests)} generation requests...")

        # Prepare prompts
        prompts = []
        gen_args = []

        for req in requests:
            prompt = req.arguments[0]
            if self.add_bos_token and self.tokenizer.bos_token:
                prompt = self.tokenizer.bos_token + prompt
            prompts.append(prompt)
            gen_args.append(req.arguments[1] if len(req.arguments) > 1 else {})

        # Run generation
        start_time = time.time()
        outputs = self.runner.generate(
            prompts,
            self.sampling_params,
            use_tqdm=not disable_tqdm,
        )
        end_time = time.time()

        total_time = end_time - start_time
        if outputs:
            self.last_e2e_total_time = float(outputs[0].get("e2e_total_time_s", 0.0) or 0.0)
            self.last_ttft = float(outputs[0].get("ttft_s", 0.0) or 0.0)
            self.last_tpot = float(outputs[0].get("tpot_s", 0.0) or 0.0)
            self.last_e2e_throughput = float(outputs[0].get("e2e_throughput_tok_s", 0.0) or 0.0)
            self.last_prefill_throughput = float(outputs[0].get("prefill_throughput_tok_s", 0.0) or 0.0)
            self.last_decode_throughput = float(outputs[0].get("decode_throughput_tok_s", 0.0) or 0.0)

        # Extract results and accumulate statistics
        results = []
        num_tokens = 0
        num_nfe = 0

        for i, output in enumerate(outputs):
            gen_kw = gen_args[i] if i < len(gen_args) else {}
            if isinstance(gen_kw, dict):
                until_raw = gen_kw.get("until")
            else:
                until_raw = getattr(gen_kw, "until", None)
            until = _normalize_until_terms(until_raw)

            trunc = output.get("text", "") or ""
            full = output.get("full_text") or trunc
            extracted = _strip_at_until_terms(trunc, until)

            self._responses_full.append(full)
            self._responses_truncated.append(trunc)
            self._responses_extracted.append(extracted)
            results.append(extracted)

            token_ids = output.get("token_ids", [])
            nfe = output.get("nfe", output.get("num_nfes", output.get("n_diff_steps", 0)))

            num_tokens += len(token_ids)
            num_nfe += nfe

            self.all_generation_times.append(total_time / len(outputs) if outputs else 0)
            self.all_nfe.append(nfe)
            self.all_tokens.append(len(token_ids))

        # Update statistics
        self.total_samples += len(requests)
        self.total_generated_tokens += num_tokens
        self.total_nfe += num_nfe
        self.total_generation_time += total_time

        # Log statistics
        if self.total_samples > 0:
            self.logger.info(
                f"Generated {len(results)} samples | "
                f"Tokens: {num_tokens} | "
                f"NFE: {num_nfe} | "
                f"Time: {total_time:.2f}s | "
                f"E2E Th: {self.last_e2e_throughput:.2f} tok/s | "
                f"Prefill Th: {self.last_prefill_throughput:.2f} tok/s | "
                f"Decode Th: {self.last_decode_throughput:.2f} tok/s"
            )

        # Save statistics if save_dir is provided
        if self.save_dir is not None:
            self._save_statistics()

        return results

    def _save_statistics(self):
        """Save statistics to file"""
        os.makedirs(self.save_dir, exist_ok=True)

        stats = {
            "total_samples": self.total_samples,
            "total_tokens": self.total_generated_tokens,
            "total_nfe": self.total_nfe,
            "total_time": self.total_generation_time,
            "e2e_total_time_s": self.last_e2e_total_time,
            "e2e_throughput_tok_s": self.last_e2e_throughput,
            "avg_tokens_per_sample": self.total_generated_tokens / self.total_samples if self.total_samples > 0 else 0,
            "avg_nfe_per_sample": self.total_nfe / self.total_samples if self.total_samples > 0 else 0,
            "avg_time_per_sample": self.total_generation_time / self.total_samples if self.total_samples > 0 else 0,
            "ttft_s": self.last_ttft,
            "tpot_s": self.last_tpot,
            "prefill_throughput_tok_s": self.last_prefill_throughput,
            "decode_throughput_tok_s": self.last_decode_throughput,
            "nfe_per_token": self.total_nfe / self.total_generated_tokens if self.total_generated_tokens > 0 else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        stats_path = os.path.join(self.save_dir, "diffulex_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Statistics saved to {stats_path}")

        if self.save_dir and self._responses_truncated:
            for fname, rows in (
                ("0x0_full_responses.json", self._responses_full),
                ("0x1_truncated_responses.json", self._responses_truncated),
                ("0x2_extracted_responses.json", self._responses_extracted),
            ):
                resp_path = os.path.join(self.save_dir, fname)
                with open(resp_path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Responses saved to {resp_path}")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuations given contexts.

        Note: This is a placeholder implementation. Full loglikelihood computation
        for diffusion models requires special handling.
        """
        self.logger.warning(
            "loglikelihood computation for diffusion models is not fully implemented. Returning placeholder values."
        )
        return [(0.0, False) for _ in requests]

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        """Compute log-likelihood of sequences."""
        raise NotImplementedError("loglikelihood_rolling is not implemented for diffusion models")
