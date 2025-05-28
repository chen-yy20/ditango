#! /bin/bash
set -x

export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODES)
export RANK=$SLURM_PROCID
export LOCAL_RANK=$(expr $SLURM_PROCID % $GPUS_PER_NODE)    

export CUDA_DEVICE_MAX_CONNECTIONS=1 # for async gradient all reduce
# export CUDA_LAUNCH_BLOCKING=1 # for debug
# export TORCH_USE_CUDA_DSA=1 # for debug

# 设置输出路径
export OUTPUT_PATH="./result/test/"

export MODEL_BASE="./ckpts"

SCRIPT="sample_video.py"
# SCRIPT="sample_video.py --ulysses-degree 8 --ring-degree 2"
# SCRIPT="preprocess.py"

ARGS="--model-base $MODEL_BASE \
--dit-weight "$MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" \
--video-size 720 1280 \
--video-length 129 \
--infer-steps 50 \
--flow-reverse \
--seed 42 \
--save-path $OUTPUT_PATH \
--vae-tiling"

# 设置日志相关
DATE=$(date +%Y%m%d_%H%M)  # 修正了日期格式，之前有错误：%Y%md
PROFILE_DIR="./logs/"
mkdir -p $PROFILE_DIR

# 创建日志文件
LOG_FILE="${PROFILE_DIR}/Hunyuan_${DATE}_${TAG}.log"

# 只有rank 0才记录日志，其他rank正常执行但不记录日志
if [ "$RANK" -eq 0 ]; then
    # rank 0: 输出重定向到日志文件
    exec python $SCRIPT $ARGS $EXTRA_ARGS 2>&1 | tee $LOG_FILE
else
    # 其他rank: 直接执行，不记录日志
    exec python $SCRIPT $ARGS $EXTRA_ARGS
fi