
import os
from RAG.LLM import DouBaoChat,Model
from TextRank import embedding_Testrank,node_Testrank
from eval_test import *
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
        self.attention_layer = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

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
        doc_pool = doc_outputs["pooler_output"]  # (batch, hidden)

        query_outputs = self.encoder(
            input_ids=query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask,
            position_ids=query_pos_ids,
        )
        query_pool = query_outputs["pooler_output"].unsqueeze(1)  # (batch, 1, hidden)

        doc_proj = self.doc_proj(doc_pool).unsqueeze(1)  # (batch, 1, hidden)
        query_proj = self.query_proj(query_pool)         # (batch, 1, hidden)

        attended, _ = self.attention_layer(query_proj, doc_proj, doc_proj)  # (batch, 1, hidden)
        attended = attended.squeeze(1)  # (batch, hidden)

        reward = self.reward_layer(attended)  # (batch, 1)
        return reward

class RewardModel_query_keshihua(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.reward_layer = nn.Linear(768, 1)
        self.query_proj = nn.Linear(768, 768)
        self.doc_proj = nn.Linear(768, 768)
        self.attention_layer = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

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
        doc_pool = doc_outputs["pooler_output"]  # (batch, hidden)

        query_outputs = self.encoder(
            input_ids=query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask,
            position_ids=query_pos_ids,
        )
        query_pool = query_outputs["pooler_output"].unsqueeze(1)  # (batch, 1, hidden)

        doc_proj = self.doc_proj(doc_pool).unsqueeze(1)  # (batch, 1, hidden)
        query_proj = self.query_proj(query_pool)         # (batch, 1, hidden)

        # 使用 MultiheadAttention：query attend to doc
        attended, _ = self.attention_layer(query_proj, doc_proj, doc_proj)  # (batch, 1, hidden)
        attended = attended.squeeze(1)  # (batch, hidden)

        reward = self.reward_layer(attended)  # (batch, 1)
        return reward



class RewardModel(nn.Module):

    def __init__(self, encoder):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone, 默认使用 ernie 3.0
        """
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
        )["pooler_output"]                              # (batch, hidden_size)
        reward = self.reward_layer(pooler_output)       # (batch, 1)
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
    encoder = AutoModel.from_pretrained('/QMKGF_master/model/ernie-3.0-base-zh')
    model = RewardModel_query(encoder)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained('/QMKGF_master/model/ernie-3.0-base-zh')

    batch_texts = [
            
        ]
    print(batch_texts)
    batch_queries = [
       
    ]

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


def pseudo_doc_llm(question):
    chat = Model()
    text = chat.chat_pseudo_doc(question, [], [])
    return text

def prompt_query_RAG(question,query,embedding,vector,reranker,num):
    
    content = vector.query(query, EmbeddingModel=embedding, k=num)
    pseudo_doc = pseudo_doc_llm(question)
    file1,file2 = content[0],content[1]
    return file1,file2 ,pseudo_doc

def prompt_RAG(question,embedding,vector,reranker):
    

    content = vector.query(question, EmbeddingModel=embedding, k=10)
    num = 0
    rerank_content = reranker.rerank(question, content, k=3)
    rerank_content = embedding_Testrank(content,question,3)
    best_content = rerank_content
    chat = Model()

    output = chat.chat_QA(question, [], best_content)

    return output

def keyword_RAG(question,lan):
    chat = Model()
    if lan == 'english':
        text = chat.chat_keyword_english(question, [], [])
        key_word_list = [item.strip() for item in text.split(',')]
        keyword = ""
        
        for key in key_word_list:
            keyword = keyword + "".join(key) + ", "

        output = question+"The key words in this question are "+"".join(keyword)
    return output

def entity_extract(question,lan):
    chat = Model()
    if lan == 'chinese':
        q_entity = chat.chat_entity_extract_chinese(question, [], [])
        output = q_entity
    elif lan == 'english':
        text = chat.chat_keyword_english(question, [], [])
        key_word_list = [item.strip() for item in text.split(',')]
        keyword = ""
        
        for key in key_word_list:
            keyword = keyword + "".join(key) + ", "

        output = question+"The key words in this question are "+"".join(keyword)
    return output