# !/usr/bin/env python3
import os
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, default_data_collator, get_scheduler

from model import RewardModel, compute_rank_list_loss , RewardModel_query
from utils import convert_example
from iTrainingLogger import iSummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='/QMKGF_master/model/ernie-3.0-base-zh', type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default='/QMKGF_master/reward/rewardtransformers/RLHF/dataset_new/hotpotQA_query_att/hotpotQA_output_80.txt', type=str, help="The path of train set.")
parser.add_argument("--dev_path", default='/QMKGF_master/reward/rewardtransformers/RLHF/dataset_new/hotpotQA_query_att/hotpotQA_output_20.txt', type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)

def evaluate_query_model(model, data_loader):

    model.eval()
    with torch.no_grad():
        batch_rank_rewards = []
        for batch in data_loader:
            for batch_idx in range(len(batch['input_ids'])):
                

                all_input_ids = batch['input_ids'][batch_idx]
                all_token_type_ids = batch['token_type_ids'][batch_idx]
                all_attention_mask = batch['attention_mask'][batch_idx]
                all_position_ids = batch['position_ids'][batch_idx]


                query_input_ids = all_input_ids[-1].unsqueeze(0).to(args.device)
                query_token_type_ids = all_token_type_ids[-1].unsqueeze(0).to(args.device)
                query_attention_mask = all_attention_mask[-1].unsqueeze(0).to(args.device)
                query_position_ids = all_position_ids[-1].unsqueeze(0).to(args.device)


                candidate_count = len(all_input_ids) - 1
                rank_rewards = []

                for text_idx in range(candidate_count):
                    candidate_input_ids = all_input_ids[text_idx].unsqueeze(0).to(args.device)
                    candidate_token_type_ids = all_token_type_ids[text_idx].unsqueeze(0).to(args.device)
                    candidate_attention_mask = all_attention_mask[text_idx].unsqueeze(0).to(args.device)
                    candidate_position_ids = all_position_ids[text_idx].unsqueeze(0).to(args.device)


                    reward = model(
                        input_ids=candidate_input_ids,
                        token_type_ids=candidate_token_type_ids,
                        attention_mask=candidate_attention_mask,
                        pos_ids=candidate_position_ids,

                        query_input_ids=query_input_ids,
                        query_token_type_ids=query_token_type_ids,
                        query_attention_mask=query_attention_mask,
                        query_pos_ids=query_position_ids
                    )

                    rank_rewards.append(reward[0])  # (1,) -> tensor([score])

                batch_rank_rewards.append(rank_rewards)
                 # (batch, rank_text_num) -> [[tensor([0.1696]), tensor([0.3466])], ...]
    model.train()
    total_ranklist, right_ranklist = 0, 0
    for rank_rewards in batch_rank_rewards:
        rank_rewards = [t.cpu().float() for t in rank_rewards]
        rank_rewards_sorted = sorted(rank_rewards, reverse=True)
        total_ranklist += 1
        if rank_rewards_sorted == rank_rewards:
            right_ranklist += 1
    return right_ranklist / total_ranklist


def train_query():
    encoder = AutoModel.from_pretrained(args.model, output_attentions=True) 

    model = RewardModel_query(encoder=encoder)# 
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    # *******************************************************************************************************
    text = "PHILIP PULLMAN [SEP] Philip Pullman is the author of Northern Lights, a young-adult fantasy novel. [SEP] NORTHERN LIGHTS"
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model(**inputs)
    attentions = outputs.attentions  
    
    attn_matrix = attentions[-1][0]   # (num_heads, seq_len, seq_len)
    head0 = attn_matrix[0]   # (seq_len, seq_len)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(tokens)
    # *******************************************************************************************************
    print(dataset)
    convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model.to(args.device)
    

    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = []
    tic_train = time.time()
    global_step, best_acc = 0, 0
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            batch_rank_rewards = []
            for batch_idx in range(len(batch['input_ids'])):
                all_input_ids = batch['input_ids'][batch_idx]
                all_token_type_ids = batch['token_type_ids'][batch_idx]
                all_attention_mask = batch['attention_mask'][batch_idx]
                all_position_ids = batch['position_ids'][batch_idx]


                query_input_ids = all_input_ids[-1].unsqueeze(0).to(args.device)
                query_token_type_ids = all_token_type_ids[-1].unsqueeze(0).to(args.device)
                query_attention_mask = all_attention_mask[-1].unsqueeze(0).to(args.device)
                query_position_ids = all_position_ids[-1].unsqueeze(0).to(args.device)

                candidate_count = len(all_input_ids) - 1
                rank_rewards = []

                for text_idx in range(candidate_count):
                    candidate_input_ids = all_input_ids[text_idx].unsqueeze(0).to(args.device)
                    candidate_token_type_ids = all_token_type_ids[text_idx].unsqueeze(0).to(args.device)
                    candidate_attention_mask = all_attention_mask[text_idx].unsqueeze(0).to(args.device)
                    candidate_position_ids = all_position_ids[text_idx].unsqueeze(0).to(args.device)

                    reward = model(
                        input_ids=candidate_input_ids,
                        token_type_ids=candidate_token_type_ids,
                        attention_mask=candidate_attention_mask,
                        pos_ids=candidate_position_ids,

                        query_input_ids=query_input_ids,
                        query_token_type_ids=query_token_type_ids,
                        query_attention_mask=query_attention_mask,
                        query_pos_ids=query_position_ids
                    )
                    rank_rewards.append(reward[0])
                batch_rank_rewards.append(rank_rewards)                # (batch, rank_text_num) -> [[tensor([0.1696]), tensor([0.3466])], ...]
            loss = compute_rank_list_loss(batch_rank_rewards, device=args.device)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                torch.save(model.state_dict(), os.path.join(cur_save_dir, 'model_state_dict.pt'))
                tokenizer.save_pretrained(cur_save_dir)
                acc = evaluate_query_model(model, eval_dataloader)
                writer.add_scalar('eval/accuracy', acc, global_step)
                writer.record()
                print("Evaluation acc: %.5f" % (acc))
                if acc > best_acc:
                    print(
                        f"best F1 performence has been updated: {best_acc:.5f} --> {acc:.5f}"
                    )
                    best_acc = acc
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    torch.save(model.state_dict(), os.path.join(cur_save_dir, 'model_state_dict.pt'))
                    tokenizer.save_pretrained(cur_save_dir)
                tic_train = time.time()


def convert_example(examples: dict, tokenizer, max_seq_len: int):
    import traceback

    import numpy as np
    from rich import print
    tokenized_output = {
        'input_ids': [], 
        'token_type_ids': [],
        'position_ids': [],
        'attention_mask': []
    }

    for example in examples['text']:
        try:
            rank_texts = example.strip().split('\t')
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            exit()

        rank_texts_prop = {
            'input_ids': [], 
            'token_type_ids': [],
            'position_ids': [],
            'attention_mask': []
        }
        for rank_text in rank_texts:
            encoded_inputs = tokenizer(
                    text=rank_text,
                    truncation=True,
                    max_length=max_seq_len,
                    padding='max_length')
            rank_texts_prop['input_ids'].append(encoded_inputs["input_ids"])
            rank_texts_prop['token_type_ids'].append(encoded_inputs["token_type_ids"])
            rank_texts_prop['position_ids'].append([i for i in range(len(encoded_inputs["input_ids"]))])
            rank_texts_prop['attention_mask'].append(encoded_inputs["attention_mask"])

        for k, v in rank_texts_prop.items():
            tokenized_output[k].append(v)
    
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)
    
    return tokenized_output

if __name__ == '__main__':
    from rich import print
    train_query()
