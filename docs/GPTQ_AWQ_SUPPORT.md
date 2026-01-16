# GPTQ/AWQ 支持

Diffulex 现在支持加载 GPTQ 和 AWQ 格式的离线量化权重，并进行推理。

## 功能概述

- **GPTQ 支持**: 支持加载 AutoGPTQ 格式的量化 checkpoint（W4A16，weight-only）
- **AWQ 支持**: 支持加载 AWQ 格式的量化 checkpoint（W4A16，weight-only）
- **离线量化**: 直接从 checkpoint 加载已量化的权重，无需先加载 bf16 再量化
- **权重缓存**: 自动缓存反量化后的权重，避免每次 forward 都重新反量化

## 使用方法

### 步骤 1: 离线量化模型（可选）

如果你有原始模型权重，可以使用 Diffulex 提供的量化脚本将其量化为 GPTQ/AWQ 格式：

```bash
# 量化模型为 GPTQ 格式
python -m diffulex.utils.quantization.quantize_model \
    --model-path /path/to/original/model \
    --output-path /path/to/output \
    --quant-format gptq \
    --group-size 128 \
    --bits 4

# 量化模型为 AWQ 格式
python -m diffulex.utils.quantization.quantize_model \
    --model-path /path/to/original/model \
    --output-path /path/to/output \
    --quant-format awq \
    --group-size 128 \
    --bits 4
```

量化脚本会生成：
- `model_quantized_{gptq|awq}.safetensors`: 包含量化权重的 safetensors 文件
- `quantization_metadata_{gptq|awq}.json`: 量化元数据

**注意**: 生成的量化权重文件需要与原始模型的配置文件（config.json）放在同一目录下，或者将量化权重文件复制到原始模型目录。

### 步骤 2: 配置和加载

在创建 `Config` 时，设置量化格式：

```python
from diffulex.config import Config

config = Config(
    model="/path/to/quantized/checkpoint",
    model_name="dream",  # 或其他模型名称
    linear_attn_weight_dtype="gptq",  # 或 "awq"
    linear_mlp_weight_dtype="gptq",   # 或 "awq"
    linear_attn_act_dtype="bf16",
    linear_mlp_act_dtype="bf16",
    tensor_parallel_size=1,  # 当前仅支持 TP=1
    # ... 其他配置
)
```

### Checkpoint 格式

#### GPTQ Checkpoint

GPTQ checkpoint 应包含以下 keys（在 `.safetensors` 文件中）：
- `{module_name}.qweight`: int8 打包的 int4 权重 [out_features, (in_features + 1) // 2]
- `{module_name}.qzeros`: int8 打包的 int4 零点 [num_groups, (in_features + 1) // 2]
- `{module_name}.scales`: float32 每组的 scales [num_groups, in_features] 或 [num_groups]
- `{module_name}.g_idx`: (可选) int32 组索引 [out_features]

#### AWQ Checkpoint

AWQ checkpoint 应包含以下 keys（在 `.safetensors` 文件中）：
- `{module_name}.qweight`: int8 打包的 int4 权重 [out_features, (in_features + 1) // 2]
- `{module_name}.qzeros`: int8 打包的 int4 零点 [num_groups, (in_features + 1) // 2]
- `{module_name}.scales`: float32 每组的 scales [num_groups, in_features] 或 [num_groups]

注意：AWQ 不使用 `g_idx`，采用顺序分组（group_id = out_idx // group_size）。

## 限制

### Tensor Parallel

当前实现仅支持 `tensor_parallel_size=1`（单 GPU）。如果使用 `tensor_parallel_size > 1`，系统会给出警告并跳过离线量化权重的加载。如果需要支持 TP>1，请提供实际的 checkpoint 以便实现 TP 切分逻辑。

### 量化格式

当前仅支持 W4A16（weight int4 + activation bf16）。不支持激活量化。

### 量化工具兼容性

- **GPTQ**: 兼容 AutoGPTQ 和 GPTQ-for-LLaMa 生成的 checkpoint
- **AWQ**: 兼容 AWQ 工具生成的 checkpoint

## 测试

### 运行单元测试

```bash
# 运行 GPTQ/AWQ 策略单元测试
pytest tests/test_gptq_awq_strategies.py -v
```

### 运行加载测试示例

```bash
# 测试 GPTQ checkpoint 加载
python examples/test_gptq_awq_loading.py \
    --format gptq \
    --model-path /path/to/gptq/checkpoint \
    --list-layers \
    --test-forward

# 测试 AWQ checkpoint 加载
python examples/test_gptq_awq_loading.py \
    --format awq \
    --model-path /path/to/awq/checkpoint \
    --list-layers \
    --test-forward
```

### 运行端到端生成测试

使用 `test_quantization_generation.py` 可以测试量化模型的完整推理流程：

```bash
# 测试 GPTQ 策略的文本生成
python examples/test_quantization_generation.py \
    --gptq \
    --model-path /path/to/quantized/model \
    --max-tokens 50

# 测试 AWQ 策略的文本生成
python examples/test_quantization_generation.py \
    --awq \
    --model-path /path/to/quantized/model \
    --max-tokens 50

# 测试特定策略组合
python examples/test_quantization_generation.py \
    --strategies gptq_w4a16_bf16kv,awq_w4a16_fp8kv \
    --model-path /path/to/quantized/model
```

### 完整工作流程示例

```bash
# 1. 量化原始模型为 GPTQ 格式
python -m diffulex.utils.quantization.quantize_model \
    --model-path /data1/ckpts/Dream-org/Dream-v0-Base-7B \
    --output-path /tmp/quantized_model \
    --quant-format gptq \
    --group-size 128 \
    --bits 4

# 2. 将量化权重复制到模型目录（或直接使用输出目录）
cp /tmp/quantized_model/model_quantized_gptq.safetensors \
   /data1/ckpts/Dream-org/Dream-v0-Base-7B/

# 3. 运行端到端测试
python examples/test_quantization_generation.py \
    --gptq \
    --model-path /data1/ckpts/Dream-org/Dream-v0-Base-7B \
    --max-tokens 50
```

## 实现细节

### 策略实现

- `LinearGPTQW4A16Strategy`: GPTQ W4A16 策略，实现 GPTQ 格式的反量化
- `LinearAWQW4A16Strategy`: AWQ W4A16 策略，实现 AWQ 格式的反量化

### 权重存储

离线量化权重存储在 `LinearBase` 的 buffers 中：
- GPTQ: `gptq_qweight`, `gptq_qzeros`, `gptq_scales`, `gptq_g_idx`
- AWQ: `awq_qweight`, `awq_qzeros`, `awq_scales`

### 前向传播

在 `LinearBase.forward()` 中：
1. 首先检查是否有离线量化权重（`has_offline_quantized_weight()`）
2. 如果有，将 GPTQ/AWQ 参数传递给 strategy 的 `linear_forward()`
3. Strategy 反量化权重（带缓存），然后使用 `F.linear()` 计算

### 加载流程

在 `load_model()` 中：
1. 首先尝试加载离线量化权重（`_load_gptq_awq_weights()`）
2. 扫描 `.safetensors` 文件中的 keys，识别 GPTQ/AWQ 格式的权重
3. 找到对应的 module，调用 `set_offline_quantized_weight()`
4. 跳过常规的 bf16 权重加载（已加载离线量化权重时）

## 性能说明

- **内存**: 离线量化权重（packed int4）显著减少内存占用
- **速度**: 当前实现使用 Python 反量化 + `F.linear()`，可能有性能开销
- **缓存**: Strategy 会缓存反量化后的权重，避免重复反量化

未来可以考虑：
- 实现 TileLang kernel 直接使用 packed 权重进行计算
- 支持更多量化格式（如 W8A16, W4A8）

## 故障排除

### 问题：无法找到模块

如果遇到 "无法找到模块" 的警告，检查：
1. Checkpoint 中的 key 命名是否与模型中的模块名称匹配
2. 如果使用 `packed_modules_mapping`，确保映射正确

### 问题：Tensor Parallel > 1

如果使用 TP>1，当前实现会跳过离线量化权重加载。解决方案：
1. 使用 TP=1（单 GPU）
2. 或提供实际的 checkpoint 以完善 TP 切分逻辑

### 问题：量化权重未加载

检查：
1. Config 中的 `linear_attn_weight_dtype` 和 `linear_mlp_weight_dtype` 是否设置为 "gptq" 或 "awq"
2. Checkpoint 是否包含必要的 keys（qweight, qzeros, scales）
3. 查看加载日志中的警告信息

## 相关文件

- `diffulex/utils/quantization/strategies/linear_gptq_w4a16.py`: GPTQ 策略实现
- `diffulex/utils/quantization/strategies/linear_awq_w4a16.py`: AWQ 策略实现
- `diffulex/layer/linear.py`: LinearBase 扩展，支持离线量化权重
- `diffulex/utils/loader.py`: 权重加载逻辑，支持 GPTQ/AWQ checkpoint
- `tests/test_gptq_awq_strategies.py`: 单元测试
- `examples/test_gptq_awq_loading.py`: 加载测试示例
