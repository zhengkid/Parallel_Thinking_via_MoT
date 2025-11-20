#!/bin/bash

# ========================================
# 1. 模型列表
# ========================================
MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen2.5-7B-Instruct"
  "/path/to/your/finetuned_model"
)

# ========================================
# 2. Seed 列表
# ========================================
SEEDS=(42 123 2025)

# ========================================
# 3. Task 列表
# ========================================
TASKS=(
  "yale-nlp/FOLIO"
  "TongZheng1999/ProofWriter"
  "TongZheng1999/ProverQA-Hard"
)

# ========================================
# 4. 固定参数
# ========================================
MODE="nl"
PROMPT_MODE="final_v2"
NUM_CANDIDATES=1
BATCH_SIZE=32
MAX_TOKENS=10000
GPU_COUNR=1
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
SPLIT="validation"

# ========================================
# 5. 循环跑所有组合
# ========================================
for MODEL_NAME_AND_PATH in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      # 自动生成唯一 Output 目录
      SAFE_MODEL_NAME=$(echo "${MODEL_NAME_AND_PATH}" | sed 's/\//_/g')
      SAFE_TASK_NAME=$(echo "${TASK}" | sed 's/\//_/g')

      OUTPUT_DIR="./results/${SAFE_MODEL_NAME}/${SAFE_TASK_NAME}/seed_${SEED}"
      mkdir -p ${OUTPUT_DIR}

      RAW_DATA_PATH="${OUTPUT_DIR}/raw_data.json"
      RESULT_PATH="${OUTPUT_DIR}/result.txt"

      echo "========================================"
      echo "Running:"
      echo "  Model: ${MODEL_NAME_AND_PATH}"
      echo "  Task:  ${TASK}"
      echo "  Seed:  ${SEED}"
      echo "  Output: ${OUTPUT_DIR}"
      echo "========================================"

      python ./eval/eval_new.py \
        --model_name_and_path ${MODEL_NAME_AND_PATH} \
        --mode ${MODE} \
        --gpu_count ${GPU_COUNR} \
        --prompt_mode ${PROMPT_MODE} \
        --dataset_name ${TASK} \
        --output_dir ${OUTPUT_DIR} \
        --save_raw_data_path ${RAW_DATA_PATH} \
        --save_result_path ${RESULT_PATH} \
        --batch_size ${BATCH_SIZE} \
        --max_tokens ${MAX_TOKENS} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --top_k ${TOP_K} \
        --seed ${SEED} \
        --split ${SPLIT} \
        --number_candidates ${NUM_CANDIDATES} \
        --use_fewshot

      echo "[FINISHED] Model=${SAFE_MODEL_NAME}, Task=${SAFE_TASK_NAME}, Seed=${SEED}"
      echo ""

    done
  done
done
