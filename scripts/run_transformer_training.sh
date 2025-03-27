#!/bin/bash

# Activate the environment
source activate /nobackup/meocakir/conda/timesnetEnv/

# List of models
MODELS=("Autoformer" "Crossformer" "DLinear" "ETSformer" "FEDformer" "FiLM" "Informer" "TimesNet" "iTransformer" "MICN" "Mamba" "Koopa" "LightTS" "MambaSimple" "MultiPatchFormer" "Nonstationary_Transformer" "PatchTST" "PAttn" "Pyraformer" "Reformer" "SegRNN" "TiDE" "TimeMixer" "TimeXer" "Transformer" "TSMixer" "WPMixer")

# Dataset information
ROOT_PATH="./dataset/transformers/"

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

seq_len=30
label_len=15


FILE_ARGS=""
for FILE in "${FILE_LIST[@]}"; do
    FILE_ARGS+=" $FILE"
done

for pred_len in 30 60 120; do
  echo "NOW RUNNING FOR pred_len:" ${pred_len}

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model Autoformer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model Crossformer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5


  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model DLinear \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model ETSformer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 2 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model FEDformer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model FiLM \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model Informer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5


  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model iTransformer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --itr 5


  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model LightTS \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5


  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model Mamba \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --expand 2 \
    --d_ff 16 \
    --d_conv 4 \
    --c_out 7 \
    --d_model 128 \
    --des 'Exp' \
    --itr 5 \

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model MICN \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model Nonstationary_Transformer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5 \
    --p_hidden_dims 256 256 \
    --p_hidden_layers 2 \
    --d_model 128


  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model PatchTST \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --n_heads 2 \
    --itr 5


  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model PAttn \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --n_heads 2 \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model Pyraformer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model Reformer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model SegRNN \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --seg_len 15 \
    --enc_in 7 \
    --d_model 512 \
    --dropout 0.5 \
    --learning_rate 0.0001 \
    --des 'Exp' \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model TiDE \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 8 \
    --d_model 256 \
    --d_ff 256 \
    --dropout 0.3 \
    --batch_size 512 \
    --learning_rate 0.1 \
    --train_epochs 10 \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model TimeMixer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --enc_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5 \
    --d_model 16 \
    --d_ff 32 \
    --learning_rate 0.01 \
    --train_epochs 10 \
    --batch_size 128 \
    --down_sampling_layers 3 \
    --down_sampling_method avg \
    --down_sampling_window 2

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model TimesNet \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --itr 5 \
    --top_k 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model TimeXer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --batch_size 4 \
    --des 'exp' \
    --itr 5


  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model Transformer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --file_list $FILE_ARGS \
    --model_id Transformer_${seq_len}_${pred_len} \
    --model TSMixer \
    --data Transformer \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 5

done