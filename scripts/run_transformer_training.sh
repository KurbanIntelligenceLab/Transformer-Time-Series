#!/bin/bash

# if it is used for the first time use this
#chmod +x run_transformer_training.sh


source /opt/miniconda3/bin/activate Transformer_TsLib


PYTHON_SCRIPT="run.py"


ROOT_PATH="./dataset/transformers/"


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


MODEL_ID="ETTh1_96_96"
MODEL="TimesNet"
DATA="Transformer"
FEATURES="M"
SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96
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


python $PYTHON_SCRIPT \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id $MODEL_ID \
    --model $MODEL \
    --data $DATA \
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

echo "Training completed!"

