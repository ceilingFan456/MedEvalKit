#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config (edit as needed)
# =========================
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODEL_NAME="ShowO"
MODEL_PATH="/home/t-qimhuang/show-o-w-clip-vit-512x512"  # local path or HF id

DATASETS_PATH="hf"
OUTPUT_ROOT="eval_results/showo"

# GPU / runtime
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
USE_VLLM="${USE_VLLM:-False}"

# Eval
SEED=42
REASONING="False"
TEST_TIMES=1

# Generation (keep small for speed; bump later)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_IMAGE_NUM="${MAX_IMAGE_NUM:-6}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

# Judge (off while validating)
USE_LLM_JUDGE="${USE_LLM_JUDGE:-False}"
JUDGE_MODEL_TYPE="openai"
GPT_MODEL="${GPT_MODEL:-gpt-4o}"
API_KEY="${AZURE_OPENAI_API_KEY:-}"
BASE_URL="${AZURE_OPENAI_ENDPOINT:-}"

# Make Show-o (and its vendored LLaVA) importable
export PYTHONPATH="$PWD:$PWD/third_party/showo:$PWD/LLaVA-NeXT"

# Optional: cap samples for quick smoke runs (your loader can read this env)
# export EVAL_MAX_SAMPLES=50

# =========================
# Datasets (small → large)
# =========================
DATASETS=(
  "VQA_RAD"
  "SLAKE"
  "PATH_VQA"
  "MedFrameQA"
  "PMC_VQA"
  "OmniMedVQA"
)

# =========================
# Helpers
# =========================
run_dataset() {
  local ds="$1"
  local out_dir="${OUTPUT_ROOT}/${ds}"
  mkdir -p "${out_dir}"

  echo "============================================================"
  echo "Dataset         : ${ds}"
  echo "Output          : ${out_dir}"
  echo "Max new tokens  : ${MAX_NEW_TOKENS}"
  [[ -n "${EVAL_MAX_SAMPLES:-}" ]] && echo "Max samples     : ${EVAL_MAX_SAMPLES}"

  # Unbuffered so tqdm updates render correctly
  python -u eval.py \
    --eval_datasets "${ds}" \
    --datasets_path "${DATASETS_PATH}" \
    --output_path "${out_dir}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --seed "${SEED}" \
    --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --use_vllm "${USE_VLLM}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_image_num "${MAX_IMAGE_NUM}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --reasoning "${REASONING}" \
    --use_llm_judge "${USE_LLM_JUDGE}" \
    --judge_model_type "${JUDGE_MODEL_TYPE}" \
    --judge_model "${GPT_MODEL}" \
    --api_key "${API_KEY}" \
    --base_url "${BASE_URL}" \
    --test_times "${TEST_TIMES}" \
    2>&1 | tee "${out_dir}/run.log"

  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "!! ${ds} failed (exit ${rc}). See ${out_dir}/run.log"
  else
    echo "-- ${ds} done."
  fi
}

# =========================
# Main
# =========================
for ds in "${DATASETS[@]}"; do
  run_dataset "${ds}"
done

echo "All VQA datasets completed (see ${OUTPUT_ROOT}/*/run.log)."
