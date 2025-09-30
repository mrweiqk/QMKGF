# !/usr/bin/env python3

import traceback

import numpy as np
from rich import print


def convert_example(examples: dict, tokenizer, max_seq_len: int):

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