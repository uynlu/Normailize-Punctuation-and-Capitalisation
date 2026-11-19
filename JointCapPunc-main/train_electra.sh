python normalization/JointCapPunc-main/main.py --model_name_or_path "FPTAI/velectra-base-discriminator-cased" \
                --model_type "electra" \
                --data_dir "normalization/data" \
                --output_dir "normalization/checkpoints/jointcappunc/electra_output" \
                --loss_coef 0.15 \
                --tuning_metric "mean_f1" \
                --max_seq_length 100 \
                --cap_emb_dim 256 \
                --train_batch_size 16 \
                --eval_batch_size 1 \
                --num_train_epochs 10 \
                --logging_steps 100 \
                --do_eval \
                --do_train \
                --use_crf \
                