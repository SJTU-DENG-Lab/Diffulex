from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _first_positive_int(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _optional_float(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        self._tokenizer_lock = threading.Lock()

        self.total_samples = 0
        self.total_generated_tokens = 0
        self.total_prompt_tokens = 0
        self.total_generation_time = 0.0
        self.last_gen_throughput = 0.0
        self.total_time_to_first_chunk = 0.0
        self.total_visible_decode_time = 0.0
        self.total_visible_decode_tokens = 0
        self.total_output_chunks = 0
        self.total_server_completion_tokens = 0
        self.total_server_steps = 0
        self.total_server_prefill_steps = 0
        self.total_server_decode_steps = 0
        self.total_server_prefill_output_tokens = 0
        self.total_server_decode_output_tokens = 0
        self.total_server_prefill_time = 0.0
        self.total_server_decode_time = 0.0
        self._responses_full: list[str] = []
        self._responses_extracted: list[str] = []
        self._sample_metrics: list[dict[str, Any]] = []
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
        with self._tokenizer_lock:
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        with self._tokenizer_lock:
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
            last_chunk_at: float | None = None
            text_pieces: list[str] = []
            chunk_token_counts: list[int] = []
            chunk_times: list[float] = []
            usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0}
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
                        usage.update(obj["usage"])
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
                    now = time.perf_counter()
                    if first_chunk_at is None:
                        first_chunk_at = now
                    last_chunk_at = now
                    text_pieces.append(delta_text)
                    chunk_token_counts.append(len(self.tok_encode(delta_text, add_special_tokens=False)))
                    chunk_times.append(now - start)
            end = time.perf_counter()
            text = "".join(text_pieces)
            time_to_first_chunk = (first_chunk_at - start) if first_chunk_at is not None else end - start
            time_to_last_chunk = (last_chunk_at - start) if last_chunk_at is not None else time_to_first_chunk
            total_time = end - start
            return text, usage, {
                "time_to_first_chunk": time_to_first_chunk,
                "time_to_last_chunk": time_to_last_chunk,
                "total_time": total_time,
                "chunk_token_counts": chunk_token_counts,
                "chunk_times_s": chunk_times,
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
                    "server_steps": _first_positive_int(
                        usage,
                        ("nfe", "num_steps", "decode_steps", "dllm_steps", "num_forward_steps"),
                    ),
                    "server_prefill_steps": _optional_float(usage, "dllm_prefill_steps"),
                    "server_decode_steps": _optional_float(usage, "dllm_decode_steps"),
                    "server_prefill_output_tokens": _optional_float(
                        usage, "dllm_prefill_output_tokens"
                    ),
                    "server_decode_output_tokens": _optional_float(
                        usage, "dllm_decode_output_tokens"
                    ),
                    "server_prefill_time_s": _optional_float(usage, "prefill_time_s"),
                    "server_decode_time_s": _optional_float(usage, "decode_time_s"),
                    "server_e2e_time_s": _optional_float(usage, "e2e_time_s"),
                    "server_tpf": _optional_float(usage, "tpf"),
                    "server_decode_tpf": _optional_float(usage, "decode_tpf"),
                    "server_decode_tps": _optional_float(usage, "decode_tps"),
                    "server_e2e_tps": _optional_float(usage, "e2e_tps"),
                    "server_aggregate_decode_tps": _optional_float(
                        usage, "aggregate_decode_tps"
                    ),
                    "server_avg_decode_tps": _optional_float(usage, "avg_decode_tps"),
                    "server_aggregate_e2e_tps": _optional_float(
                        usage, "aggregate_e2e_tps"
                    ),
                    "server_avg_e2e_tps": _optional_float(usage, "avg_e2e_tps"),
                    "server_time_semantics": usage.get("dllm_time_semantics"),
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
                "server_steps": _first_positive_int(
                    usage,
                    ("nfe", "num_steps", "decode_steps", "dllm_steps", "num_forward_steps"),
                ),
                "server_prefill_steps": _optional_float(usage, "dllm_prefill_steps"),
                "server_decode_steps": _optional_float(usage, "dllm_decode_steps"),
                "server_prefill_output_tokens": _optional_float(
                    usage, "dllm_prefill_output_tokens"
                ),
                "server_decode_output_tokens": _optional_float(
                    usage, "dllm_decode_output_tokens"
                ),
                "server_prefill_time_s": _optional_float(usage, "prefill_time_s"),
                "server_decode_time_s": _optional_float(usage, "decode_time_s"),
                "server_e2e_time_s": _optional_float(usage, "e2e_time_s"),
                "server_tpf": _optional_float(usage, "tpf"),
                "server_decode_tpf": _optional_float(usage, "decode_tpf"),
                "server_decode_tps": _optional_float(usage, "decode_tps"),
                "server_e2e_tps": _optional_float(usage, "e2e_tps"),
                "server_aggregate_decode_tps": _optional_float(
                    usage, "aggregate_decode_tps"
                ),
                "server_avg_decode_tps": _optional_float(usage, "avg_decode_tps"),
                "server_aggregate_e2e_tps": _optional_float(
                    usage, "aggregate_e2e_tps"
                ),
                "server_avg_e2e_tps": _optional_float(usage, "avg_e2e_tps"),
                "server_time_semantics": usage.get("dllm_time_semantics"),
            },
            "timing": timing,
        }

    def _generate_one_request(self, req: Instance) -> dict:
        with record_function(f"{self.engine_name}.client.generate_one"):
            prompt = self._format_prompt(req.arguments[0])
            gen_kwargs = req.arguments[1] if len(req.arguments) > 1 else {}
            until_terms = _normalize_until_terms(
                gen_kwargs.get("until") if isinstance(gen_kwargs, dict) else None
            )
            if isinstance(gen_kwargs, dict) and "max_gen_toks" in gen_kwargs:
                max_new_tokens = min(int(gen_kwargs["max_gen_toks"]), self.max_new_tokens)
            else:
                max_new_tokens = self.max_new_tokens
            return self._completion_request(prompt, until_terms, max_new_tokens)

    def generate_until(self, requests_list: List[Instance], disable_tqdm: bool = True):
        del disable_tqdm
        outputs = []
        if self.profile_session.enabled and not self.profile_session.active:
            self.profile_session.start()
        batch_size = max(int(self.batch_size_per_gpu), 1)
        for start in range(0, len(requests_list), batch_size):
            batch = requests_list[start : start + batch_size]
            if batch_size == 1:
                batch_outputs = [self._generate_one_request(req) for req in batch]
            else:
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    batch_outputs = list(executor.map(self._generate_one_request, batch))
            outputs.extend(batch_outputs)
            for _ in batch_outputs:
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
            server_steps = meta.get("server_steps")
            server_steps = int(server_steps) if server_steps is not None else None
            server_prefill_steps = meta.get("server_prefill_steps")
            server_prefill_steps = int(server_prefill_steps) if server_prefill_steps is not None else None
            server_decode_steps = meta.get("server_decode_steps")
            server_decode_steps = int(server_decode_steps) if server_decode_steps is not None else None
            server_prefill_output_tokens = meta.get("server_prefill_output_tokens")
            server_prefill_output_tokens = (
                int(server_prefill_output_tokens) if server_prefill_output_tokens is not None else None
            )
            server_decode_output_tokens = meta.get("server_decode_output_tokens")
            server_decode_output_tokens = (
                int(server_decode_output_tokens) if server_decode_output_tokens is not None else None
            )
            server_prefill_time = meta.get("server_prefill_time_s")
            server_decode_time = meta.get("server_decode_time_s")
            server_e2e_time = meta.get("server_e2e_time_s")
            server_tpf = meta.get("server_tpf")
            server_decode_tpf = meta.get("server_decode_tpf")
            server_decode_tps = meta.get("server_decode_tps")
            server_e2e_tps = meta.get("server_e2e_tps")
            server_aggregate_decode_tps = (
                meta.get("server_aggregate_decode_tps") or server_decode_tps
            )
            server_avg_decode_tps = meta.get("server_avg_decode_tps")
            server_aggregate_e2e_tps = (
                meta.get("server_aggregate_e2e_tps") or server_e2e_tps
            )
            server_avg_e2e_tps = meta.get("server_avg_e2e_tps")
            has_server_dllm_metrics = (
                server_steps is not None
                and server_prefill_time is not None
                and server_decode_time is not None
            )
            time_to_first_chunk = float(timing["time_to_first_chunk"])
            time_to_last_chunk = float(timing.get("time_to_last_chunk", time_to_first_chunk))
            total_time = float(timing["total_time"])
            visible_decode_tokens = max(completion_tokens - first_chunk_tokens, 0)
            visible_decode_time = max(time_to_last_chunk - time_to_first_chunk, 0.0)
            request_tail_time = max(total_time - time_to_last_chunk, 0.0)
            output_chunks = len(chunk_token_counts)
            chunk_tpf = completion_tokens / output_chunks if output_chunks > 0 else 0.0
            tpf = server_tpf if server_tpf is not None else (completion_tokens / server_steps if server_steps else None)
            chunk_intervals = [
                max(float(b) - float(a), 0.0)
                for a, b in zip(timing.get("chunk_times_s", [])[:-1], timing.get("chunk_times_s", [])[1:])
            ]
            first_chunk_visible_tps = first_chunk_tokens / time_to_first_chunk if time_to_first_chunk > 0 else 0.0
            visible_decode_tps = visible_decode_tokens / visible_decode_time if visible_decode_time > 0 else 0.0
            e2e_tps = completion_tokens / total_time if total_time > 0 else 0.0
            if has_server_dllm_metrics:
                self.total_server_prefill_time += float(server_prefill_time)
                self.total_server_decode_time += float(server_decode_time)
                self.total_server_prefill_steps += server_prefill_steps or 0
                self.total_server_decode_steps += server_decode_steps or 0
                self.total_server_prefill_output_tokens += server_prefill_output_tokens or 0
                self.total_server_decode_output_tokens += server_decode_output_tokens or 0
            self._responses_full.append(text)
            self._responses_extracted.append(extracted)
            self.total_prompt_tokens += prompt_tokens
            self.total_generated_tokens += completion_tokens
            self.total_server_completion_tokens += server_completion_tokens
            self.total_time_to_first_chunk += time_to_first_chunk
            self.total_visible_decode_time += visible_decode_time
            self.total_visible_decode_tokens += visible_decode_tokens
            self.total_output_chunks += output_chunks
            self.total_server_steps += server_steps or 0
            self.total_generation_time += total_time
            self._sample_metrics.append(
                {
                    "metric_semantics": "openai_stream_visible_chunks",
                    "server_metric_semantics": (
                        "server_dllm_usage" if has_server_dllm_metrics else None
                    ),
                    "server_time_semantics": meta.get("server_time_semantics"),
                    "prefill_time_s": server_prefill_time,
                    "decode_time_s": server_decode_time,
                    "ttft_s": time_to_first_chunk,
                    "time_to_first_chunk_s": time_to_first_chunk,
                    "time_to_last_chunk_s": time_to_last_chunk,
                    "request_tail_time_s": request_tail_time,
                    "total_time_s": total_time,
                    "ptps": (
                        prompt_tokens / server_prefill_time
                        if server_prefill_time and server_prefill_time > 0
                        else None
                    ),
                    "dtps": (
                        server_aggregate_decode_tps
                        if server_aggregate_decode_tps is not None
                        else visible_decode_tps
                    ),
                    "decode_tps": server_aggregate_decode_tps,
                    "aggregate_decode_tps": server_aggregate_decode_tps,
                    "avg_decode_tps": server_avg_decode_tps,
                    "decode_tpf": server_decode_tpf,
                    "visible_decode_tps": visible_decode_tps,
                    "first_chunk_visible_tps": first_chunk_visible_tps,
                    "e2e_tps": e2e_tps,
                    "server_e2e_tps": server_aggregate_e2e_tps,
                    "aggregate_server_e2e_tps": server_aggregate_e2e_tps,
                    "avg_server_e2e_tps": server_avg_e2e_tps,
                    "server_e2e_time_s": server_e2e_time,
                    "tpf": tpf,
                    "chunk_tpf": chunk_tpf,
                    "prompt_tokens": float(prompt_tokens),
                    "completion_tokens": float(completion_tokens),
                    "first_chunk_tokens": float(first_chunk_tokens),
                    "visible_decode_tokens": float(visible_decode_tokens),
                    "streamed_completion_tokens": float(streamed_completion_tokens),
                    "num_output_chunks": float(output_chunks),
                    "chunk_token_counts": chunk_token_counts,
                    "chunk_times_s": timing.get("chunk_times_s", []),
                    "chunk_intervals_s": chunk_intervals,
                    "server_prompt_tokens": float(server_prompt_tokens),
                    "server_completion_tokens": float(server_completion_tokens),
                    "server_steps": float(server_steps) if server_steps else None,
                    "server_prefill_steps": (
                        float(server_prefill_steps) if server_prefill_steps is not None else None
                    ),
                    "server_decode_steps": (
                        float(server_decode_steps) if server_decode_steps is not None else None
                    ),
                    "server_prefill_output_tokens": (
                        float(server_prefill_output_tokens)
                        if server_prefill_output_tokens is not None
                        else None
                    ),
                    "server_decode_output_tokens": (
                        float(server_decode_output_tokens)
                        if server_decode_output_tokens is not None
                        else None
                    ),
                }
            )
            results.append(extracted)

        self.total_samples += len(outputs)
        if self.save_dir:
            self._save_statistics()
        if self.profile_session.enabled and self.profile_session.steps >= self.profile_session.active_steps > 0:
            self.profile_session.stop()
        return results

    def _save_statistics(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        e2e_tps_values = [
            float(m["e2e_tps"])
            for m in self._sample_metrics
            if float(m.get("total_time_s", 0.0)) > 0 and float(m.get("completion_tokens", 0.0)) > 0
        ]
        avg_e2e_tps = _mean(e2e_tps_values)
        visible_decode_tps_values = [
            float(m["visible_decode_tps"])
            for m in self._sample_metrics
            if float(m.get("visible_decode_tokens", 0.0)) > 0
            and float(m.get("time_to_last_chunk_s", 0.0)) > float(m.get("time_to_first_chunk_s", 0.0))
        ]
        chunk_tpf_values = [
            float(m["chunk_tpf"])
            for m in self._sample_metrics
            if float(m.get("num_output_chunks", 0.0)) > 0 and float(m.get("completion_tokens", 0.0)) > 0
        ]
        tpf_values = [
            float(m["tpf"])
            for m in self._sample_metrics
            if m.get("tpf") is not None
        ]
        decode_tpf_values = [
            float(m["decode_tpf"])
            for m in self._sample_metrics
            if m.get("decode_tpf") is not None
        ]
        server_decode_tps_values = [
            float(m["avg_decode_tps"] if m.get("avg_decode_tps") is not None else m["decode_tps"])
            for m in self._sample_metrics
            if m.get("decode_tps") is not None
        ]
        server_e2e_tps_values = [
            float(
                m["avg_server_e2e_tps"]
                if m.get("avg_server_e2e_tps") is not None
                else m["server_e2e_tps"]
            )
            for m in self._sample_metrics
            if m.get("server_e2e_tps") is not None
        ]
        has_server_dllm_metrics = self.total_server_steps > 0 and self.total_server_decode_time > 0
        has_server_decode_output_tokens = self.total_server_decode_output_tokens > 0
        server_decode_tokens = (
            self.total_server_decode_output_tokens
            if has_server_decode_output_tokens
            else self.total_generated_tokens
        )
        client_wall_e2e_tps = (
            self.total_generated_tokens / self.total_generation_time
            if self.total_generation_time > 0
            else 0.0
        )
        aggregate_decode_tps = (
            server_decode_tokens / self.total_server_decode_time
            if has_server_dllm_metrics
            else None
        )
        aggregate_visible_decode_tps = (
            self.total_visible_decode_tokens / self.total_visible_decode_time
            if self.total_visible_decode_time > 0
            else 0.0
        )
        aggregate_server_e2e_tps = (
            self.total_generated_tokens
            / (self.total_server_prefill_time + self.total_server_decode_time)
            if has_server_dllm_metrics
            else None
        )
        aggregate_e2e_tps = (
            aggregate_server_e2e_tps
            if aggregate_server_e2e_tps is not None
            else client_wall_e2e_tps
        )
        avg_e2e_tps_for_report = (
            _mean(server_e2e_tps_values)
            if has_server_dllm_metrics and server_e2e_tps_values
            else avg_e2e_tps
        )
        stats = {
            "engine_name": self.engine_name,
            "metric_semantics": (
                "server_dllm_usage" if has_server_dllm_metrics else "openai_stream_visible_chunks"
            ),
            "metric_notes": {
                "e2e_tps_tok_s": (
                    "legacy alias for aggregate_e2e_tps_tok_s. Uses server prefill+decode time when "
                    "DLLM server usage is available; otherwise uses client wall-clock time."
                ),
                "decode_tps_tok_s": (
                    "legacy alias for aggregate_decode_tps_tok_s; for DLLM server metrics this uses decode-phase "
                    "output tokens, not tokens accepted during prefill."
                ),
                "aggregate_tps": (
                    "total tokens divided by server prefill+decode time when DLLM server usage is available; "
                    "otherwise total tokens divided by client wall-clock time."
                ),
                "avg_tps": "arithmetic mean of per-request TPS values.",
                "visible_decode_tps_tok_s": (
                    "visible output tokens after the first non-empty chunk divided by elapsed time until the "
                    "last non-empty chunk; this is chunk-stream throughput, not true model decode throughput."
                ),
                "tpf": (
                    "not available unless the server returns DLLM NFE/step count; chunk_tpf is visible tokens "
                    "per emitted chunk."
                ),
            },
            "total_samples": self.total_samples,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_generated_tokens,
            "total_server_completion_tokens": self.total_server_completion_tokens or None,
            "total_time": self.total_generation_time,
            "total_time_to_first_chunk_s": self.total_time_to_first_chunk,
            "total_visible_decode_time_s": self.total_visible_decode_time,
            "total_visible_decode_tokens": self.total_visible_decode_tokens,
            "total_output_chunks": self.total_output_chunks,
            "total_server_steps": self.total_server_steps or None,
            "total_server_prefill_steps": self.total_server_prefill_steps or None,
            "total_server_decode_steps": self.total_server_decode_steps or None,
            "total_server_prefill_output_tokens": (
                self.total_server_prefill_output_tokens if has_server_decode_output_tokens else None
            ),
            "total_server_decode_output_tokens": (
                self.total_server_decode_output_tokens if has_server_decode_output_tokens else None
            ),
            "total_prefill_time_s": self.total_server_prefill_time if has_server_dllm_metrics else None,
            "total_decode_time_s": self.total_server_decode_time if has_server_dllm_metrics else None,
            "e2e_tps_tok_s": aggregate_e2e_tps,
            "aggregate_e2e_tps_tok_s": aggregate_e2e_tps,
            "avg_e2e_tps_tok_s": avg_e2e_tps_for_report,
            "client_wall_e2e_tps_tok_s": client_wall_e2e_tps,
            "avg_client_e2e_tps_tok_s": avg_e2e_tps,
            "avg_ttft_s": self.total_time_to_first_chunk / self.total_samples if self.total_samples > 0 else 0.0,
            "avg_time_to_first_chunk_s": (
                self.total_time_to_first_chunk / self.total_samples if self.total_samples > 0 else 0.0
            ),
            "last_gen_throughput_tok_s": self.last_gen_throughput,
            "ptps_tok_s": (
                self.total_prompt_tokens / self.total_server_prefill_time
                if has_server_dllm_metrics and self.total_server_prefill_time > 0
                else None
            ),
            "dtps_tok_s": (
                aggregate_decode_tps
                if aggregate_decode_tps is not None
                else aggregate_visible_decode_tps
            ),
            "visible_decode_tps_tok_s": aggregate_visible_decode_tps,
            "aggregate_visible_decode_tps_tok_s": aggregate_visible_decode_tps,
            "avg_visible_decode_tps_tok_s": _mean(visible_decode_tps_values),
            "decode_tps_tok_s": aggregate_decode_tps,
            "aggregate_decode_tps_tok_s": aggregate_decode_tps,
            "avg_decode_tps_tok_s": _mean(server_decode_tps_values) if server_decode_tps_values else None,
            "server_e2e_tps_tok_s": aggregate_server_e2e_tps,
            "aggregate_server_e2e_tps_tok_s": aggregate_server_e2e_tps,
            "avg_server_e2e_tps_tok_s": _mean(server_e2e_tps_values) if server_e2e_tps_values else None,
            "tpf": self.total_generated_tokens / self.total_server_steps if self.total_server_steps > 0 else None,
            "decode_tpf": (
                self.total_server_decode_output_tokens / self.total_server_decode_steps
                if self.total_server_decode_steps > 0 and has_server_decode_output_tokens
                else None
            ),
            "avg_tpf": _mean(tpf_values) if tpf_values else None,
            "avg_decode_tpf": _mean(decode_tpf_values) if decode_tpf_values else None,
            "chunk_tpf": (
                self.total_generated_tokens / self.total_output_chunks if self.total_output_chunks > 0 else 0.0
            ),
            "avg_chunk_tpf": _mean(chunk_tpf_values),
            "aggregate_tps": aggregate_e2e_tps,
            "aggregate_tps_tok_s": aggregate_e2e_tps,
            "avg_tps": avg_e2e_tps_for_report,
            "avg_tps_tok_s": avg_e2e_tps_for_report,
            "avg_e2e_tps": avg_e2e_tps_for_report,
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
