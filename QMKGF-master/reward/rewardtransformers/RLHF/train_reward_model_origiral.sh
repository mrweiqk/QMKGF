
python train_reward_model.py \
    --model "/QMKGF-master/model/bert-base-chinese" \
    --train_path "/QMKGF-master/data/train_k=10_last.tsv" \
    --dev_path "/QMKGF-master/data/dev_k=10_last.tsv" \
    --save_dir "checkpoints/reward_model/bert_kg_k=10_reward_3row" \
    --img_log_dir "logs/reward_model/bert_kg_k=10_reward_3row" \
    --img_log_name "BERT Reward Model" \
    --batch_size 4 \
    --max_seq_len 512 \
    --learning_rate 1e-5 \
    --valid_steps 100 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"