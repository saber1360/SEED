export CUDA_VISIBLE_DEVICES=5

model_name=SEED
seq_len=96

for pred_len in 96
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_len 16 \
  --dropout 0.8 \
  --top_p 0.0 \
  --des 'Exp' \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 15 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1 \
  --use_revin True \
  --use_dropout 0.05
done

for pred_len in 192
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_len 16 \
  --dropout 0.6 \
  --alpha 0.5 \
  --des 'Exp' \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 15 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1 \
  --use_revin True \
  --use_dropout 0.1
done

for pred_len in 336
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_len 24 \
  --alpha 0.3 \
  --dropout 0.5 \
  --top_p 0.0 \
  --des 'Exp' \
  --learning_rate 0.0003 \
  --batch_size 32 \
  --train_epochs 15 \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --use_revin True \
  --use_dropout 0.1 \
  --patience 10
done


for pred_len in 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id ETTh2_new_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_len 16 \
  --alpha 0.5 \
  --dropout 0.4 \
  --top_p 0.0 \
  --des 'Exp' \
  --learning_rate 0.0003 \
  --batch_size 32 \
  --train_epochs 15 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1 \
  --use_revin True \
  --use_dropout 0.1 \
  --patience 10
done