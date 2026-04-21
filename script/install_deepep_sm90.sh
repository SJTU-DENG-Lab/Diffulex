#!/usr/bin/env bash
set -euo pipefail

git submodule update --init --recursive

VENV_ROOT="$(pwd)"
echo "[VENV_ROOT]: $VENV_ROOT"

NVSHMEM_ROOT="$VENV_ROOT/.venv/lib/python3.12/site-packages/nvidia/nvshmem"
NVSHMEM_LIB_DIR="$NVSHMEM_ROOT/lib"

echo "[NVSHMEM_ROOT]: $NVSHMEM_ROOT"
echo "[NVSHMEM_LIB_DIR]: $NVSHMEM_LIB_DIR"

if [[ ! -d "$NVSHMEM_LIB_DIR" ]]; then
  echo "ERROR: NVSHMEM lib dir not found: $NVSHMEM_LIB_DIR"
  exit 1
fi

cd "$NVSHMEM_LIB_DIR"

if [[ -L "libnvshmem_host.so" ]]; then
  echo "[INFO] libnvshmem_host.so is a symlink, removing it"
  rm -f "libnvshmem_host.so"
elif [[ -e "libnvshmem_host.so" ]]; then
  echo "[INFO] libnvshmem_host.so exists and is not a symlink, keeping it"
else
  echo "[INFO] libnvshmem_host.so does not exist, will create it"
fi

if [[ ! -e "libnvshmem_host.so" ]]; then
  if [[ -e "libnvshmem_host.so.3" ]]; then
    ln -s "libnvshmem_host.so.3" "libnvshmem_host.so"
    echo "[INFO] linked libnvshmem_host.so -> libnvshmem_host.so.3"
  else
    target="$(find . -maxdepth 1 -type f -name 'libnvshmem_host.so.*' | sort | head -n 1 | sed 's|^\./||')"
    if [[ -n "${target:-}" ]]; then
      ln -s "$target" "libnvshmem_host.so"
      echo "[INFO] linked libnvshmem_host.so -> $target"
    else
      echo "ERROR: no libnvshmem_host.so.* found in $NVSHMEM_LIB_DIR"
      exit 1
    fi
  fi
fi

echo "[INFO] current host libs:"
ls -l libnvshmem_host.so*

cd "$VENV_ROOT/thirdparty/DeepEP"

unset TORCH_CUDA_ARCH_LIST
export TORCH_CUDA_ARCH_LIST="9.0"
unset DISABLE_SM90_FEATURES

rm -rf build
find . -name '*.so' -delete 2>/dev/null || true
 
export NVSHMEM_DIR="$NVSHMEM_ROOT"
export LD_LIBRARY_PATH="$NVSHMEM_LIB_DIR:${LD_LIBRARY_PATH:-}"

echo "[INFO] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "[INFO] NVSHMEM_DIR=$NVSHMEM_DIR"
echo "[INFO] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

python setup.py install

python -c "import deep_ep; print('<<< deep_ep import ok >>>')"