import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(embeddings):
    embeddings = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


def initialize_weights(similarity_matrix, core_sentence_index, alpha=0.85):
    n = len(similarity_matrix)
    weights = np.ones(n) * (1 - alpha)
    weights[core_sentence_index] = alpha
    return weights


def compute_rank_embedding(similarity_matrix, top_k=3):
    G = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(G, weight='weight')
    top_k_indices = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return top_k_indices


def compute_rank_node(similarity_matrix, weights, top_k=3):
    G = nx.from_numpy_array(similarity_matrix)
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            if similarity_matrix[i, j] > 0:
                edge_weight = np.sqrt(weights[i] * weights[j]) * similarity_matrix[i, j]
                G[i][j]['weight'] = edge_weight
                G[j][i]['weight'] = edge_weight
    scores = nx.pagerank(G, weight='weight')
    top_k_indices = sorted(scores, key=scores.get, reverse=True)[1:top_k+1]
    return top_k_indices


def find_related_sentences_node(Q_embedding, K_embeddings, top_k=3):
    sent_emb = K_embeddings.tolist()
    core_emb = Q_embedding.tolist()
    sentences_with_core = sent_emb + [core_emb]
    similarity_matrix = calculate_similarity(sentences_with_core)
    core_sentence_index = len(sent_emb)
    weights = initialize_weights(similarity_matrix, core_sentence_index, alpha=0.85)
    ranked_sentences = compute_rank_node(similarity_matrix, weights, top_k)
    return ranked_sentences


def find_related_sentences_embedding(Q_embedding, K_embeddings, top_k=3):
    similarity_matrix = calculate_similarity((Q_embedding + K_embeddings).tolist())
    ranked_sentences = compute_rank_embedding(similarity_matrix, top_k)
    return ranked_sentences


def attn_textrank(Q_embedding, K_embeddings, top_k):
    num_iterations = 120
    damping_factor = 0.65
    num_nodes = K_embeddings.shape[0]
    similarities = np.dot(K_embeddings, Q_embedding) / (np.linalg.norm(K_embeddings, axis=1) * np.linalg.norm(Q_embedding))
    graph = np.zeros((num_nodes + 1, num_nodes + 1))
    graph[0, 1:] = similarities
    graph[1:, 0] = similarities
    scores = np.zeros(num_nodes + 1)
    scores[0] = 1
    for _ in range(num_iterations):
        new_scores = np.zeros(num_nodes + 1)
        for i in range(num_nodes + 1):
            for j in range(num_nodes + 1):
                if i != j:
                    new_scores[i] += graph[i, j] * scores[j]
        scores = (1 - damping_factor) * scores + damping_factor * new_scores
    top_indices = np.argsort(scores[1:])[::-1][:top_k]
    return top_indices


def emb_textrank(Q_embedding, K_embeddings, top_k):
    embedding = [Q_embedding.tolist()] + K_embeddings.tolist()
    num_node = len(embedding)
    cosine_matrix = cosine_similarity(embedding)
    G = nx.Graph()
    for i in range(num_node):
        G.add_node(i)
    for i in range(1, num_node):
        similarity = cosine_matrix[0][i]
        if similarity > 0:
            G.add_edge(0, i, weight=similarity)
    scores = nx.pagerank(G, weight='weight')
    scores.pop(0)
    top_k_indices = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [i-1 for i in top_k_indices]


if __name__ == '__main__':
    Q = ""
    K = []
    K = [i.replace("\n", "") for i in K]

    model = SentenceTransformer('/llm_models/m3e', device="cuda")
    Q_embedding = model.encode(Q)
    K_embeddings = model.encode(K)

    index1 = attn_textrank(Q_embedding, K_embeddings, 3)
    for id in index1:
        print(K[id])
