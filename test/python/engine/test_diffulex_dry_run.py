import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest
import torch
import yaml

from diffulex import SamplingParams

# Run each test in forked process to avoid torch.distributed double-init / leak between tests.
# Skip in CI by default (no GPU/checkpoints); run locally with: pytest --forked
pytestmark = [
    pytest.mark.forked,
    pytest.mark.diffulex_dry_run,
    pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip diffulex dry-run in CI (GPU + checkpoints required)",
    ),
]


_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "test_diffulex_dry_run.yaml"
OUTPUT_BASE = Path(__file__).resolve().parent / "output" / "test_diffulex_dry_run"
MAX_RUNS_RETAINED = 10

with open(_CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

CKPT = Path(CONFIG["ckpt"])
GSM8K_NUM_SAMPLES = CONFIG["gsm8k_num_samples"]
CHECKPOINT_RELS = {k: tuple(v) for k, v in CONFIG["checkpoint_rels"].items()}
STRATEGY_CONFIG = {k: tuple(v) for k, v in CONFIG["strategy_config"].items()}
DECODING_THRESHOLDS = CONFIG["decoding_thresholds"]
ENGINE_KWARGS = CONFIG["engine"]
SAMPLING_PARAMS = CONFIG["sampling"]
FEW_SHOT_BASE = CONFIG["few_shot_base"]
FEW_SHOT_INSTRUCT = CONFIG["few_shot_instruct"]


def _ensure_output_run_dir() -> Path:
    """Create timestamped run dir, prune old runs, return path."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_BASE / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    dirs = sorted(OUTPUT_BASE.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs[MAX_RUNS_RETAINED:]:
        if d.is_dir():
            shutil.rmtree(d)
    return run_dir


def _save_outputs(output_path: Path, test_name: str, strategy: str, outputs: list) -> None:
    payload = {"test": test_name, "strategy": strategy, "outputs": outputs}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _ckpt(rel: str | None) -> str | None:
    """Resolve relative path under CKPT."""
    return str(CKPT / rel) if rel else None


def _get_model_config(name: str):
    """(model_path, lora_path, model_name, decoding_strategy, buffer_size, few_shot, few_shot_type)."""
    m_rel, l_rel = CHECKPOINT_RELS[name]
    cfg = STRATEGY_CONFIG[name]
    dec, buf = cfg[0], cfg[1]
    few_shot_type = cfg[2] if len(cfg) > 2 else "instruct"
    few_shot = FEW_SHOT_BASE if few_shot_type == "base" else FEW_SHOT_INSTRUCT
    return _ckpt(m_rel), _ckpt(l_rel), name, dec, buf, few_shot, few_shot_type


def build_prompts(questions, prefix="", few_shot=None, few_shot_type="instruct"):
    """Build prompts from questions, prefix, few_shot text, and format (base|instruct)."""
    few_shot = few_shot or FEW_SHOT_INSTRUCT
    if few_shot_type == "base":
        suffix = lambda q: f"\n\nQ: {q}\nA: "
    else:
        suffix = lambda q: f"<|im_start|>user\nQuestion: {q}\nAnswer:<|im_end|>\n<|im_start|>assistant\n"
    return [prefix + few_shot + suffix(q) for q in questions]


def _run_diffulex_test(
    model,
    model_name,
    decoding_strategy,
    prompts,
    sampling_params,
    use_lora=False,
    lora_path=None,
    buffer_size=4,
    save_output_path: Path | None = None,
    test_name: str = "",
    **kwargs,
):
    """Shared runner for Diffulex dry-run tests."""
    from diffulex import Diffulex

    common_kwargs = dict(ENGINE_KWARGS, buffer_size=buffer_size, decoding_thresholds=DECODING_THRESHOLDS)
    common_kwargs.update(kwargs)

    llm_kwargs = dict(
        model_name=model_name,
        decoding_strategy=decoding_strategy,
        **common_kwargs,
    )
    if use_lora and lora_path:
        llm_kwargs["use_lora"] = True
        llm_kwargs["lora_path"] = lora_path

    llm = Diffulex(model, **llm_kwargs)
    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    if save_output_path is not None:
        _save_outputs(save_output_path, test_name, model_name, outputs)
    return outputs


def get_gsm8k_prompts(strategy_name: str):
    """Build GSM8K prompts for the given strategy (uses its few_shot_type and tokenizer)."""
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")
    dataset = datasets.load_dataset("gsm8k", "main")["test"]["question"][:GSM8K_NUM_SAMPLES]
    model_path = _ckpt(CHECKPOINT_RELS[strategy_name][0])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prefix = tokenizer.bos_token or ""
    _, _, _, _, _, few_shot, few_shot_type = _get_model_config(strategy_name)
    return build_prompts(dataset, prefix, few_shot, few_shot_type)


@pytest.fixture
def sampling_params():
    return SamplingParams(**SAMPLING_PARAMS)


@pytest.fixture(scope="session")
def dry_run_output_dir():
    return _ensure_output_run_dir()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_d2f_llada(sampling_params, dry_run_output_dir, request):
    name = "llada"
    model, lora, _, dec, buf, _, _ = _get_model_config(name)
    prompts = get_gsm8k_prompts(name)
    _run_diffulex_test(
        model,
        model_name=name,
        decoding_strategy=dec,
        prompts=prompts,
        sampling_params=sampling_params,
        use_lora=True,
        lora_path=lora,
        buffer_size=buf,
        save_output_path=dry_run_output_dir / f"{request.node.name}.json",
        test_name=request.node.name,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_d2f_dream(sampling_params, dry_run_output_dir, request):
    name = "dream"
    model, lora, _, dec, buf, _, _ = _get_model_config(name)
    prompts = get_gsm8k_prompts(name)
    _run_diffulex_test(
        model,
        model_name=name,
        decoding_strategy=dec,
        prompts=prompts,
        sampling_params=sampling_params,
        use_lora=True,
        lora_path=lora,
        buffer_size=buf,
        save_output_path=dry_run_output_dir / f"{request.node.name}.json",
        test_name=request.node.name,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sdar(sampling_params, dry_run_output_dir, request):
    name = "sdar"
    model, lora, _, dec, buf, _, _ = _get_model_config(name)
    prompts = get_gsm8k_prompts(name)
    _run_diffulex_test(
        model,
        model_name=name,
        decoding_strategy=dec,
        prompts=prompts,
        sampling_params=sampling_params,
        buffer_size=buf,
        save_output_path=dry_run_output_dir / f"{request.node.name}.json",
        test_name=request.node.name,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fastdllmv2(sampling_params, dry_run_output_dir, request):
    name = "fast_dllm_v2"
    model, lora, _, dec, buf, _, _ = _get_model_config(name)
    prompts = get_gsm8k_prompts(name)
    _run_diffulex_test(
        model,
        model_name=name,
        decoding_strategy=dec,
        prompts=prompts,
        sampling_params=sampling_params,
        buffer_size=buf,
        save_output_path=dry_run_output_dir / f"{request.node.name}.json",
        test_name=request.node.name,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("name", list(CHECKPOINT_RELS))
def test_diffulex_strategies_parametrized(name, sampling_params, dry_run_output_dir, request):
    """Parametrized coverage over all strategies."""
    model, lora, model_name, dec, buf, _, _ = _get_model_config(name)
    prompts = get_gsm8k_prompts(name)
    _run_diffulex_test(
        model,
        model_name=model_name,
        decoding_strategy=dec,
        prompts=prompts,
        sampling_params=sampling_params,
        use_lora=(lora is not None),
        lora_path=lora,
        buffer_size=buf,
        save_output_path=dry_run_output_dir / f"{request.node.name}.json",
        test_name=request.node.name,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
