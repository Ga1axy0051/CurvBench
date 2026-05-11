#!/bin/bash

# =================================================================
# 实验配置和搜索空间 (Node Classification - NC)
# =================================================================

# 目标模型和数据集
MODELS=("HGCN")

# 你的目标数据集
# 建议：Cora/Pubmed 和 Airport/Disease 最好分开跑，因为它们的最佳参数区间完全不重叠
DATASETS=("citeseer" ) 

MANIFOLD=("PoincareBall")
#'lr': 0.007944363302120835, 'dropout': 0.5, 'weight_decay': 0.0013226021274709864, 'c': '1.0', 'act': 'None'
# === 关键修正 1: 针对 Airport/Disease 的搜索空间 ===
# Disease/Airport 这种结构性数据集，LR 0.01 容易炸，建议向下搜
LRS=("0.007944363302120835" ) 

# Disease 通常不需要 Dropout，Airport 需要很小
DROPOUTS=(" 0.5" ) 

# 维度：HGCN 在低维优势大，16 足够；高维反而难优化
DIMS=("16") 

# === 关键修正 2: Weight Decay ===
# 双曲空间对 WD 很敏感，Airport/Disease 建议搜 0
WEIGHT_DECAYS=("0.0013226021274709864" )

NUM_LAYERS=("2")
ACTIVATIONS=("None") 
#pumbed 0.01 0.5 1e-2 
# === 关键修正 3: 曲率 c ===
# None = 可学习; 1.0 = 固定;
# Airport/Disease 如果不稳，固定为 1.0 通常有奇效
C_VALUES=("1.0") 

# 种子控制和硬件
SPLIT_SEED=1234        

NUM_RUNS=5   # 调试阶段跑5次，确定参数后再跑10次        
BASE_SEED=0            
GPU_ID=1             

# 结果保存
LOG_BASE_DIR="Result"
mkdir -p "$LOG_BASE_DIR"

echo "Starting HGCN Grid Search with Double Precision"

# =================================================================
# 主循环
# =================================================================

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for lr in "${LRS[@]}"; do
            for drp in "${DROPOUTS[@]}"; do
                for dim in "${DIMS[@]}"; do
                    for wd in "${WEIGHT_DECAYS[@]}"; do
                        for n_layers in "${NUM_LAYERS[@]}"; do 
                            for act in "${ACTIVATIONS[@]}"; do
                                for c_val in "${C_VALUES[@]}"; do

                                    # 处理 c 参数
                                    if [ "$c_val" == "None" ]; then
                                        C_ARG="--c None"
                                        C_NAME="Learnable"
                                    else
                                        C_ARG="--c $c_val"
                                        C_NAME="$c_val"
                                    fi
                            
                                    # 确定当前超参数组合的唯一目录
                                    PARAM_ID="lr${lr}_drp${drp}_wd${wd}_c${C_NAME}"
                                    BASE_RUN_DIR="${LOG_BASE_DIR}/${model}_${dataset}/${PARAM_ID}"
                                    
                                    echo "--- Params: ${dataset} | LR:${lr} | WD:${wd} | C:${C_NAME} ---"
                                    
                                    for i in $(seq 1 $NUM_RUNS); do
                                        CURRENT_SEED=$((BASE_SEED + i - 1))
                                        
                                        python train.py \
                                            --dataset ${dataset} \
                                            --task nc \
                                            --model ${model} \
                                            --manifold ${MANIFOLD} \
                                            --lr ${lr} \
                                            --dropout ${drp} \
                                            --dim ${dim} \
                                            --weight-decay ${wd} \
                                            --num-layers ${n_layers} \
                                            --act ${act} \
                                            --bias 1 \
                                            --grad-clip 1\
                                            --cuda ${GPU_ID} \
                                            --seed ${CURRENT_SEED} \
                                            --split-seed ${SPLIT_SEED} \
                                            --epochs 2000 \
                                            --patience 100 \
                                            --optimizer RiemannianAdam\
                                            --use-feats 1\
                                            --log-freq 5 \
                                            --c $c_val \
                                            --save 1 \
                                            --save-dir "${BASE_RUN_DIR}/seed_${CURRENT_SEED}" > /dev/null
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Grid Search Finished."