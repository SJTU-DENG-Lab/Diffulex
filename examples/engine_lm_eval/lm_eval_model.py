from __future__ import annotations

import atexit
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import requests
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer

from diffulex.profiling import TorchProfileSession, record_function

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


def _normalize_until_terms(until: object) -> list[str]:
    if until is None:
        return []
    if isinstance(until, str):
        return [until] if until else []
    if isinstance(until, (list, tuple)):
        return [str(x) for x in until if x is not None and str(x) != ""]
    return []


def _strip_at_until_terms(response: str, until_terms: list[str]) -> str:
    out = response
    for term in until_terms:
        if term:
            out = out.split(term)[0]
    return out


@register_model("engine_oai")
class EngineOpenAILM(LM):
    def __init__(
        self,
        base_url: str,
        model: str,
        engine_name: str = "sglang",
        api_key: str = "EMPTY",
        tokenizer_path: Optional[str] = None,
        batch_size: int | str = 1,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        ignore_eos: bool = False,
        add_bos_token: bool = False,
        apply_chat_template: bool = False,
        chat_completions: bool = False,
        timeout: float = 600.0,
        verify: bool = True,
        save_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **_,
    ) -> None:
        super().__init__()
        self.engine_name = engine_name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.batch_size_per_gpu = int(batch_size)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.ignore_eos = ignore_eos
        self.add_bos_token = add_bos_token
        self.apply_chat_template = apply_chat_template
        self.chat_completions = chat_completions
        self.timeout = timeout
        self.verify = verify
        self.save_dir = save_dir
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model,
            trust_remote_code=trust_remote_code,
        )

        self.total_samples = 0
        self.total_generated_tokens = 0
        self.total_prompt_tokens = 0
        self.total_generation_time = 0.0
        self.last_gen_throughput = 0.0
        self.total_ttft = 0.0
        self.total_decode_time = 0.0
        self._responses_full: list[str] = []
        self._responses_extracted: list[str] = []
        self._sample_metrics: list[dict[str, float]] = []
        self.profile_session = TorchProfileSession(f"{self._file_prefix}_client")
        atexit.register(self.profile_session.stop)

    @property
    def _file_prefix(self) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in self.engine_name) or "engine"

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return "remote"

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    @classmethod
    def create_from_arg_string(cls: Type[T], arg_string: str, additional_config: Optional[dict] = None) -> T:
        additional_config = additional_config or {}
        args = utils.simple_parse_args_string(arg_string)
        args.update({k: v for k, v in additional_config.items() if v is not None and k not in args})
        return cls(**args)

    @classmethod
    def create_from_arg_obj(cls: Type[T], arg_dict: dict, additional_config: Optional[dict] = None) -> T:
        additional_config = additional_config or {}
        args = dict(arg_dict)
        args.update({k: v for k, v in additional_config.items() if v is not None and k not in args})
        return cls(**args)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(text, add_special_tokens=add_special_tokens).input_ids

    def _format_prompt(self, prompt: str) -> str:
        if self.apply_chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        if self.add_bos_token and self.tokenizer.bos_token:
            return self.tokenizer.bos_token + prompt
        return prompt

    def _server_info(self) -> dict:
        try:
            resp = requests.get(
                f"{self.base_url}/get_server_info",
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify,
            )
            if resp.status_code == 404:
                return {}
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return {}

    def _stream_openai_chunks(self, url: str, payload: Dict[str, Any]) -> tuple[str, dict, dict]:
        with record_function(f"{self.engine_name}.client.http_stream"):
            start = time.perf_counter()
            first_chunk_at: float | None = None
            text_pieces: list[str] = []
            chunk_token_counts: list[int] = []
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            with requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
                verify=self.verify,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data:"):
                        continue
                    line = raw_line[len("data:") :].strip()
                    if line == "[DONE]":
                        break
                    obj = json.loads(line)
                    if "error" in obj:
                        raise RuntimeError(f"Engine streaming error: {obj['error']}")
                    if "usage" in obj and obj["usage"] is not None:
                        usage["prompt_tokens"] = obj["usage"].get("prompt_tokens", usage["prompt_tokens"])
                        usage["completion_tokens"] = obj["usage"].get("completion_tokens", usage["completion_tokens"])
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta_text = ""
                    if "text" in choice:
                        delta_text = choice.get("text") or ""
                    elif "delta" in choice:
                        delta_text = choice.get("delta", {}).get("content") or ""
                    if not delta_text:
                        continue
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    text_pieces.append(delta_text)
                    chunk_token_counts.append(len(self.tok_encode(delta_text, add_special_tokens=False)))
            end = time.perf_counter()
            text = "".join(text_pieces)
            ttft = (first_chunk_at - start) if first_chunk_at is not None else end - start
            total_time = end - start
            return text, usage, {
                "ttft": ttft,
                "total_time": total_time,
                "chunk_token_counts": chunk_token_counts,
            }

    def _completion_request(self, prompt: str, until_terms: list[str], max_new_tokens: int) -> dict:
        if self.chat_completions:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": max_new_tokens,
                "stop": until_terms or None,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            text, usage, timing = self._stream_openai_chunks(
                f"{self.base_url}/v1/chat/completions",
                payload,
            )
            return {
                "text": text,
                "prompt": prompt,
                "meta_info": {
                    "server_prompt_tokens": usage.get("prompt_tokens", 0),
                    "server_completion_tokens": usage.get("completion_tokens", 0),
                },
                "timing": timing,
            }

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": max_new_tokens,
            "stop": until_terms or None,
            "ignore_eos": self.ignore_eos,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        text, usage, timing = self._stream_openai_chunks(
            f"{self.base_url}/v1/completions",
            payload,
        )
        return {
            "text": text,
            "prompt": prompt,
            "meta_info": {
                "server_prompt_tokens": usage.get("prompt_tokens", 0),
                "server_completion_tokens": usage.get("completion_tokens", 0),
            },
            "timing": timing,
        }

    def generate_until(self, requests_list: List[Instance], disable_tqdm: bool = False):
        del disable_tqdm
        outputs = []
        for req in requests_list:
            self.profile_session.start()
            with record_function(f"{self.engine_name}.client.generate_one"):
                prompt = self._format_prompt(req.arguments[0])
                gen_kwargs = req.arguments[1] if len(req.arguments) > 1 else {}
                until_terms = _normalize_until_terms(gen_kwargs.get("until") if isinstance(gen_kwargs, dict) else None)
                max_new_tokens = (
                    int(gen_kwargs.get("max_gen_toks", self.max_new_tokens))
                    if isinstance(gen_kwargs, dict)
                    else self.max_new_tokens
                )
                outputs.append(self._completion_request(prompt, until_terms, max_new_tokens))
            self.profile_session.step()

        info = self._server_info()
        if info.get("internal_states"):
            self.last_gen_throughput = info["internal_states"][0].get("last_gen_throughput", 0.0)

        results = []
        for output, req in zip(outputs, requests_list):
            gen_kwargs = req.arguments[1] if len(req.arguments) > 1 else {}
            until_terms = _normalize_until_terms(gen_kwargs.get("until") if isinstance(gen_kwargs, dict) else None)
            text = output["text"]
            prompt = output["prompt"]
            extracted = _strip_at_until_terms(text, until_terms)
            meta = output["meta_info"]
            timing = output["timing"]
            prompt_tokens = len(self.tok_encode(prompt, add_special_tokens=False))
            completion_tokens = len(self.tok_encode(text, add_special_tokens=False))
            chunk_token_counts = [int(x) for x in timing.get("chunk_token_counts", [])]
            streamed_completion_tokens = sum(chunk_token_counts)
            if chunk_token_counts:
                first_chunk_tokens = min(chunk_token_counts[0], completion_tokens)
            else:
                first_chunk_tokens = min(completion_tokens, 1 if completion_tokens > 0 else 0)
            server_prompt_tokens = int(meta.get("server_prompt_tokens", 0))
            server_completion_tokens = int(meta.get("server_completion_tokens", 0))
            ttft = float(timing["ttft"])
            total_time = float(timing["total_time"])
            decode_tokens = max(completion_tokens - first_chunk_tokens, 0)
            decode_time = max(total_time - ttft, 0.0)
            ptps = prompt_tokens / ttft if ttft > 0 else 0.0
            dtps = decode_tokens / decode_time if decode_time > 0 else 0.0
            e2e_tps = completion_tokens / total_time if total_time > 0 else 0.0
            self._responses_full.append(text)
            self._responses_extracted.append(extracted)
            self.total_prompt_tokens += prompt_tokens
            self.total_generated_tokens += completion_tokens
            self.total_ttft += ttft
            self.total_decode_time += decode_time
            self.total_generation_time += total_time
            self._sample_metrics.append(
                {
                    "prefill_time_s": ttft,
                    "decode_time_s": decode_time,
                    "ttft_s": ttft,
                    "total_time_s": total_time,
                    "ptps": ptps,
                    "dtps": dtps,
                    "e2e_tps": e2e_tps,
                    "prompt_tokens": float(prompt_tokens),
                    "completion_tokens": float(completion_tokens),
                    "first_chunk_tokens": float(first_chunk_tokens),
                    "streamed_completion_tokens": float(streamed_completion_tokens),
                    "chunk_token_counts": chunk_token_counts,
                    "server_prompt_tokens": float(server_prompt_tokens),
                    "server_completion_tokens": float(server_completion_tokens),
                }
            )
            results.append(extracted)

        self.total_samples += len(outputs)
        if self.save_dir:
            self._save_statistics()
        if self.profile_session.steps >= self.profile_session.active_steps > 0:
            self.profile_session.stop()
        return results

    def _save_statistics(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        stats = {
            "engine_name": self.engine_name,
            "total_samples": self.total_samples,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_generated_tokens,
            "total_time": self.total_generation_time,
            "total_prefill_time_s": self.total_ttft,
            "total_decode_time_s": self.total_decode_time,
            "e2e_tps_tok_s": self.total_generated_tokens / self.total_generation_time if self.total_generation_time > 0 else 0.0,
            "avg_ttft_s": self.total_ttft / self.total_samples if self.total_samples > 0 else 0.0,
            "last_gen_throughput_tok_s": self.last_gen_throughput,
            "ptps_tok_s": self.total_prompt_tokens / self.total_ttft if self.total_ttft > 0 else 0.0,
            "dtps_tok_s": (
                sum(max(int(m["completion_tokens"]) - int(m["first_chunk_tokens"]), 0) for m in self._sample_metrics)
                / self.total_decode_time
                if self.total_decode_time > 0
                else 0.0
            ),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        Path(self.save_dir, f"{self._file_prefix}_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        Path(self.save_dir, f"{self._file_prefix}_sample_metrics.json").write_text(
            json.dumps(self._sample_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        Path(self.save_dir, f"{self._file_prefix}_full_responses.json").write_text(
            json.dumps(self._responses_full, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        Path(self.save_dir, f"{self._file_prefix}_extracted_responses.json").write_text(
            json.dumps(self._responses_extracted, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        eval_logger.warning("loglikelihood is not implemented for EngineOpenAILM.")
        return [(0.0, False) for _ in requests]

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError("loglikelihood_rolling is not implemented for EngineOpenAILM")
