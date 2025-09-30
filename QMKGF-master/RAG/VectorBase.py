#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional, Tuple, Union
import json
from RAG.Embeddings import BaseEmbeddings, OpenAIEmbedding, JinaEmbedding, ZhipuEmbedding
import numpy as np
from tqdm import tqdm

class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        self.document = document
        self.entity_document = []
    
    def get_list_vector(self, EmbeddingModel: BaseEmbeddings,entity_list) -> List[List[float]]:
        self.entity_vectors = []
        self.entity_document = entity_list
        for doc in tqdm(entity_list):
            self.entity_vectors.append(EmbeddingModel.get_embedding(doc))
        return self.entity_vectors

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors
    
    def entity_persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/entity_doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.entity_document, f, ensure_ascii=False)
        if self.entity_vectors:
            with open(f"{path}/entity_vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.entity_vectors, f)

    def persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def load_entity_vector(self, path: str = 'storage'):
        with open(f"{path}/entity_vectors.json", 'r', encoding='utf-8') as f:
            self.entity_vectors = json.load(f)
        with open(f"{path}/entity_doecment.json", 'r', encoding='utf-8') as f:
            self.entity_document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)
    
    def entity_query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1, returnall = 0) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector)
                          for vector in self.entity_vectors])
        document_array = np.array(self.entity_document)
        sorted_indices = result.argsort()
        if returnall : 
            k = len(self.entity_vectors)
        last_k_indices = sorted_indices[-k:]
        reversed_last_k_indices = last_k_indices[::-1]
        selected_elements = document_array[reversed_last_k_indices]
        final_result = selected_elements.tolist()
        return final_result

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1, returnall = 0) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector)
                          for vector in self.vectors])
        document_array = np.array(self.document)
        sorted_indices = result.argsort()
        if returnall : 
            k = len(self.vectors)
        last_k_indices = sorted_indices[-k:]
        reversed_last_k_indices = last_k_indices[::-1]
        selected_elements = document_array[reversed_last_k_indices] 
        final_result = selected_elements.tolist()
        return final_result
