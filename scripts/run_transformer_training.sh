#!/bin/bash

# Activate the environment
source activate /nobackup/meocakir/conda/timesnetEnv/

# List of models
MODELS=("Autoformer" "Crossformer" "DLinear" "ETSformer" "FEDformer" "FiLM" "Informer" "TimesNet" "iTransformer" "MICN" "Mamba" "Koopa" "LightTS" "MambaSimple" "MultiPatchFormer" "Nonstationary_Transformer" "PatchTST" "PAttn" "Pyraformer" "Reformer" "SegRNN" "TiDE" "TimeMixer" "TimeXer" "Transformer" "TSMixer" "WPMixer")

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

FILE_ARGS=""
for FILE in "${FILE_LIST[@]}"; do
    FILE_ARGS+=" $FILE"
done

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model Autoformer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model Crossformer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model DLinear \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model ETSformer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model FEDformer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model FiLM \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model Informer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model iTransformer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model Koopa \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model LightTS \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model Mamba \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 7 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model MICN \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model MultiPatchFormer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model Nonstationary_Transformer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --d_model 128


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model PatchTST \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model PAttn \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model Pyraformer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model Reformer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model SegRNN \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --seg_len 24 \
  --enc_in 7 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model TiDE \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
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
  --patience 5 \
  --train_epochs 10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model TimeMixer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 16 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --train_epochs 10 \
  --patience 10 \
  --batch_size 128 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model TimesNet \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model TimeXer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --batch_size 4 \
  --des 'exp' \
  --itr 1
  

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model Transformer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --file_list $FILE_ARGS \
  --model_id Transformer_30_30 \
  --model TSMixer \
  --data Transformer \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1