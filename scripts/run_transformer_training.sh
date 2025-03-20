#!/bin/bash

# Activate the environment
source /opt/miniconda3/bin/activate Transformer_TsLib

# List of models
MODELS=("Autoformer" "Crossformer" "DLinear" "ETSformer" "FEDformer" "FiLM" "Informer" "TimesNet" "iTransformer" "MICN" "Mamba" "Koopa" "LightTS" "MambaSimple" "MultiPatchFormer" "Nonstationary_Transformer" "PatchTST" "PAttn" "Pyraformer" "Reformer" "SegRNN" "TiDE" "TimeMixer" "TimeXer" "Transformer" "TSMixer" "WPMixer")

# can add different parameters to change the forecast length
SEQ_LENGTHS=(30)
LABEL_LENGTHS=(15)
PREDICTION_LENGTHS=(30)

# Dataset information
ROOT_PATH="./dataset/transformers/"
PYTHON_SCRIPT="run.py"

# File list for the dataset
FILE_LIST=(
    "transformer_C_part_1.csv"
    "transformer_C_part_2.csv"
    "transformer_D_part_1.csv"
    "transformer_D_part_2.csv"
    "transformer_E_part_1.csv"
    "transformer_E_part_2.csv"
    "transformer_F_part_1.csv"
    "transformer_F_part_2.csv"
    "transformer_F_part_3.csv"
    "transformer_F_part_4.csv"
    "transformer_G.csv"
    "transformer_H.csv"
    "transformer_I_part_1.csv"
    "transformer_I_part_2.csv"
    "transformer_J_part_1.csv"
    "transformer_J_part_2.csv"
)

# Common hyperparameters
FEATURES="M"
E_LAYERS=2
D_LAYERS=1
FACTOR=3
ENC_IN=7
DEC_IN=7
C_OUT=7
D_MODEL=8
D_FF=16
TOP_K=5
TRAIN_EPOCHS=1

FILE_ARGS=""
for FILE in "${FILE_LIST[@]}"; do
    FILE_ARGS+=" $FILE"
done

for MODEL in "${MODELS[@]}"; do
    echo "Running $MODEL"

    for i in "${!PREDICTION_LENGTHS[@]}"; do
        SEQ_LEN=${SEQ_LENGTHS[i]}
        LABEL_LEN=${LABEL_LENGTHS[i]}
        PRED_LEN=${PREDICTION_LENGTHS[i]}

        MODEL_ID="ETTh1_${SEQ_LEN}_${PRED_LEN}"
        echo "️Running $MODEL with SEQ_LEN = $SEQ_LEN, LABEL_LEN = $LABEL_LEN, PRED_LEN = $PRED_LEN"

        python $PYTHON_SCRIPT \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $ROOT_PATH \
            --file_list $FILE_ARGS \
            --model_id $MODEL_ID \
            --model $MODEL \
            --data Transformer \
            --features $FEATURES \
            --seq_len $SEQ_LEN \
            --label_len $LABEL_LEN \
            --pred_len $PRED_LEN \
            --e_layers $E_LAYERS \
            --d_layers $D_LAYERS \
            --factor $FACTOR \
            --enc_in $ENC_IN \
            --dec_in $DEC_IN \
            --c_out $C_OUT \
            --d_model $D_MODEL \
            --d_ff $D_FF \
            --des Exp \
            --itr 1 \
            --top_k $TOP_K \
            --train_epochs $TRAIN_EPOCHS

        echo "Completed $MODEL with SEQ_LEN = $SEQ_LEN, LABEL_LEN = $LABEL_LEN, PRED_LEN = $PRED_LEN"
    done
done

echo "All models have finished training!"
