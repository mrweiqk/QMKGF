import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel_query(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.reward_layer = nn.Linear(768, 1)
        self.query_proj = nn.Linear(768, 768)
        self.doc_proj = nn.Linear(768, 768)
        self.attention_layer = nn.MultiheadAttention(embed_dim=768, num_heads=64, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        pos_ids: torch.Tensor = None,
        query_input_ids: torch.Tensor = None,
        query_token_type_ids: torch.Tensor = None,
        query_attention_mask: torch.Tensor = None,
        query_pos_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        doc_outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )
        doc_pool = doc_outputs["pooler_output"]

        query_outputs = self.encoder(
            input_ids=query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask,
            position_ids=query_pos_ids,
        )
        query_pool = query_outputs["pooler_output"].unsqueeze(1)

        doc_proj = self.doc_proj(doc_pool).unsqueeze(1)
        query_proj = self.query_proj(query_pool)

        attended, _ = self.attention_layer(query_proj, doc_proj, doc_proj)
        attended = attended.squeeze(1)

        reward = self.reward_layer(attended)
        return reward


class RewardModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.reward_layer = nn.Linear(768, 1)

    def forward(
        self,
        input_ids: torch.tensor,
        token_type_ids: torch.tensor,
        attention_mask=None,
        pos_ids=None,
    ) -> torch.tensor:
        pooler_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )["pooler_output"]
        reward = self.reward_layer(pooler_output)
        return reward


def compute_rank_list_loss(rank_rewards_list: List[List[torch.tensor]], device='cpu') -> torch.Tensor:
    if type(rank_rewards_list) != list:
        raise TypeError(f'@param rank_rewards expected "list", received {type(rank_rewards)}.')
    
    loss, add_count = torch.tensor([0]).to(device), 0
    for rank_rewards in rank_rewards_list:
        for i in range(len(rank_rewards)-1):
            for j in range(i+1, len(rank_rewards)):
                diff = F.logsigmoid(rank_rewards[i] - rank_rewards[j])
                loss = loss + diff
                add_count += 1
    loss = loss / add_count
    return -loss


if __name__ == '__main__':
    from rich import print
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AutoModel.from_pretrained('/model/ernie-3.0-base-zh')
    model = RewardModel_query(encoder)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained('/model/ernie-3.0-base-zh')

    batch_texts = []
    print(batch_texts)
    batch_queries = []

    rank_rewards = []

    for texts, query in zip(batch_texts, batch_queries):
        query_inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=128).to(device)
        query_inputs = {k: v.to(device) for k, v in query_inputs.items()}

        tmp = []
        for text in texts:
            doc_inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
            doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}

            r = model(
                input_ids=doc_inputs["input_ids"],
                token_type_ids=doc_inputs["token_type_ids"],
                attention_mask=doc_inputs["attention_mask"],
                query_input_ids=query_inputs["input_ids"],
                query_token_type_ids=query_inputs["token_type_ids"],
                query_attention_mask=query_inputs["attention_mask"],
            )
            tmp.append(r[0].to(device))
        rank_rewards.append(tmp)

    print('rank_rewards: ', rank_rewards)

    loss = compute_rank_list_loss(rank_rewards, device=device)
    print('loss: ', loss)

    loss.backward()

    print(model)
