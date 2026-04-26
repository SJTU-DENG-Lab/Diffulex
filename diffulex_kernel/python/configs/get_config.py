import os
import shutil
import vllm.model_executor.layers.fused_moe as fm

src = os.path.join(
    os.path.dirname(os.path.realpath(fm.__file__)),
    "configs",
    "E=128,N=512,device_name=NVIDIA_H100_80GB_HBM3.json",
)
dst = "diffulex_kernel/python/configs/E=128,N=512,device_name=NVIDIA_H100_80GB_HBM3.json"

print("src:", src)
print("exists:", os.path.exists(src))
if not os.path.exists(src):
    raise SystemExit("vLLM package does not contain this H100 MoE config; need tune manually or copy from another env.")

shutil.copyfile(src, dst)
print("copied ->", dst)