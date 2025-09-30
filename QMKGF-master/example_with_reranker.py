import os
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import DouBaoChat, Model
from RAG.Embeddings import BgeEmbedding
from RAG.Reranker import BgeReranker
from reward.rewardtransformers.RLHF.inference_reward_model import *
import json
import time
from tqdm import tqdm
from eval_test import *
from RAG.TextRank_new import attn_textrank
from sentence_transformers import SentenceTransformer
from model import *
from RAG.BM25.BM25 import rank_bm25
from visual_kg import *
from ppr import *
import os
import torch
from model import RewardModel
from rich import print
import pandas as pd
from transformers import AutoTokenizer, AutoModel, ErnieModel, BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

itor = 1000
list_embedding = 0
have_created_db = True
judge_tf = False

datasets = "iirc"
emb_model = "bge_large"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = ''

if emb_model == "bge_large":
    path = '/model/bge-large-zh-v1.5'

model = SentenceTransformer(path, device="cuda")

print(f"------------------datasets = {datasets}-------------------------")
print(f"------------------emb_model = {emb_model}-------------------------")
print(f"------------------path = {path}-------------------------")

def replace_empty_or_none(s):
    result = []
    for item in s:
        if item == '[]':
            result.append('Empty')
        elif item is None:
            result.append('Empty')
        elif item == "":
            result.append('Empty')
        else:
            result.append(item)
    return result

def model_eval(answer_list, perd_answer_list, perd_answer_num_list, lan, judge_tf=False):
    eval = eval_test()
    perd_answer_list = replace_empty_or_none(perd_answer_list)
    if judge_tf:
        eval.calculate_em(answer_list, perd_answer_list, lan)
        return 
    eval.rouge_eval(answer_list, perd_answer_list, lan)
    eval.bleu_eval(answer_list, perd_answer_list, lan)
    eval.meteor_eval(answer_list, perd_answer_list, lan)

def RAG(question, doc_list, reranker):
    rerank_content = reranker.rerank(question, doc_list, k=3)
    best_content = rerank_content
    chat = Model()
    output = chat.chat_QA(question, [], best_content)
    return output

def kg_ext(kg, question):
    entity_list, relation_list, triple_list = [], [], []
    entity_list = set()
    relation_list = []
    triple_list = []
    for triplet in kg:
        entity1, relation, entity2 = triplet
        entity_list.add(entity1)
        entity_list.add(entity2)
        relation_list.append(relation)
        triple_list.append(f"({entity1}, {relation}, {entity2})")
    entities_str = ", ".join([f"{question}{entity}" for entity in entity_list])
    relations_str = " | ".join([f"{question}{relation}" for relation in relation_list])
    triplets_str = " | ".join([f"{question}{triplet}" for triplet in triple_list])
    return entities_str, relations_str, triplets_str

def deduplicate_list(input_list):
    seen = set()
    deduped_list = []
    for item in input_list:
        if item not in seen:
            deduped_list.append(item)
            seen.add(item)
    return deduped_list

def get_rank(candidates, ground_truth):
    try:
        return candidates.index(ground_truth) + 1
    except ValueError:
        return 100

def kg_retrieve_chat(ans, vector, embedding, kg, question, reranker, lan, content, judge_tf, k=3):
    entity_list, relation_list, triple_list = kg_ext(kg, question)
    question_content = vector.query(question, EmbeddingModel=embedding, k=10)
    entity_content = vector.query(entity_list, EmbeddingModel=embedding, k=3)
    relation_content = vector.query(relation_list, EmbeddingModel=embedding, k=3)
    triple_content = vector.query(triple_list, EmbeddingModel=embedding, k=3)
    merged_list = question_content + entity_content + relation_content + triple_content
    kg_rerank_content = reranker.rerank(question, merged_list, k=len(merged_list))
    kg_num = get_rank(kg_rerank_content, ans)
    kg_rerank_content = reranker.rerank(question, merged_list, k=3)
    chat = Model()
    if judge_tf:
        kg_output = chat.chat_QA_judge_tf(question, [], kg_rerank_content)
    elif lan == 'chinese':
        kg_output = chat.chat_QA_chinese(question, [], kg_rerank_content)
    elif lan == 'english':
        kg_output = chat.chat_QA_english(question, [], kg_rerank_content)
    return kg_num, kg_output, kg_rerank_content

def entity_map(q_entity, entity_list, embedding, vector):
    if q_entity in entity_list:
        return q_entity
    entity_list_vec = vector.entity_query(q_entity, embedding)
    return entity_list_vec[0]

def cal_sim_score(score, score1, score2, score3, score4):
    scores = [score1, score2, score3, score4]
    valid_scores = [s for s in scores if s != 0 and s != 100]
    if not valid_scores:
        return None
    valid_sum = sum(valid_scores)
    return score / valid_sum

def create_kg(question, G, model, Q_entity, k, lan, sel=0):
    time2 = 0
    time3 = 0
    time2_start = time.time()
    subgraph, _ = personal_pagerank_subgraph(G, Q_entity, k)
    subgraph = subgraph2list(subgraph, Q_entity)
    One_BFS_neighbors = One_BFS(G, Q_entity)
    One_DFS_graph = One_DFS(G, Q_entity, One_BFS_neighbors)
    sim_k_list = sim_k(model, k, Q_entity, One_BFS_neighbors)
    one_bfs_k_list = One_BFS_k(sim_k_list, k)
    one_dfs_k_list = One_DFS_k(One_DFS_graph, k)
    subgraph_relation_list = create_triple(G, subgraph, Q_entity, lan)
    if sel == 1:
        return subgraph_relation_list
    sim_relation_list = create_triple(G, sim_k_list, Q_entity, lan)
    dfs_relation_list = create_triple(G, one_dfs_k_list, Q_entity, lan)
    time2_end = time.time()
    time2 = time2 + (time2_end - time2_start)
    time3_start = time.time()
    lastkg_index = kg_score2(subgraph_relation_list, sim_relation_list, dfs_relation_list, 1)
    time3_end = time.time()
    time3 = time3 + (time3_end - time3_start)
    if lastkg_index == 0:
        return subgraph_relation_list, sim_relation_list, dfs_relation_list, time2, time3
    elif lastkg_index == 1:
        return sim_relation_list, subgraph_relation_list, dfs_relation_list, time2, time3
    elif lastkg_index == 2:
        return dfs_relation_list, sim_relation_list, subgraph_relation_list, time2, time3
    return 

def kg_score2(line1, line2, line3, re=1):
    if re == 1:
        if datasets == "iirc":
            model_dir = '/QMKGF-master/model/ernie-3.0-base-zh'
            model_ckpt = '/QMKGF-master/reward/rewardtransformers/RLHF/checkpoints/reward_model/iirc/ernie_kg_k=10_reward_3row/model_best/model_state_dict.pt'
        elif datasets == "musique":
            model_dir = '/QMKGF-master/model/ernie-3.0-base-zh'
            model_ckpt = '/QMKGF-master/reward/rewardtransformers/RLHF/checkpoints/reward_model/musique/ernie_kg_k=10_reward_3row/model_best/model_state_dict.pt'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = AutoModel.from_pretrained(model_dir)
    model = RewardModel(encoder)
    model.load_state_dict(torch.load(model_ckpt, map_location=device), strict=False)
    model.to(device)
    model.eval()
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    formatted_str = str(line1).replace('"', '').replace("'", "").replace("[", "").replace("]", "")
    result1 = f"{formatted_str}"
    formatted_str = str(line2).replace('"', '').replace("'", "").replace("[", "").replace("]", "")
    result2 = f"{formatted_str}"
    formatted_str = str(line3).replace('"', '').replace("'", "").replace("[", "").replace("]", "")
    result3 = f"{formatted_str}"
    texts = [result1, result2, result3]
    inputs = tokenizer(
        texts, 
        max_length=512,
        padding='max_length',  
        truncation=True,      
        return_tensors='pt'  
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad(): 
        r = model(**inputs)
    best_index = torch.argmax(r).item() 
    return best_index

def kg_fusion(list1, list2, list3, question, embedding_model):
    list1_text = " ".join([" ".join(triple) for triple in list1])
    q_vec = np.array(embedding_model.get_embedding(question)).reshape(1, -1)
    list1_vec = np.array(embedding_model.get_embedding(list1_text)).reshape(1, -1)
    base_sim = cosine_similarity(q_vec, list1_vec)[0][0]
    for triple_list in [list2, list3]:
        for triple in triple_list:
            triple_text = " ".join(triple)
            triple_vec = np.array(embedding_model.get_embedding(triple_text)).reshape(1, -1)
            sim = cosine_similarity(q_vec, triple_vec)[0][0]
            if sim > base_sim:
                list1.append(triple)
    return list1

def protect_empty_string(st):
    result = re.sub(r'(?:\[\])+', 'Empty', st)
    return result

def all_RAG(datasets, embedding, vector, reranker, question_list, answer_list, lan, judge_tf=False):
    pred_answer_kg = []
    pred_answer_kg_ref, pred_answer_tradition_ref, pred_answer_base_ref, pred_answer_bm25_ref = [], [], [], []
    query, pred_answer_kg_llm_rerank_ref = [], []
    pred_answer_kg_num, pred_answer_tradition_num, pred_answer_base_num = [], [], []
    random.seed(42)
    time1, time2, time3, time4 = 0, 0, 0, 0
    kg_time = 0
    ans_num = 0
    test_num = 0
    if datasets == "iirc":
        file_path = '/QMKGF-master/data/iirc/triples.jsonl'
        if emb_model == "bge_large":
            model = SentenceTransformer('/QMKGF-master/model/bge-large-zh-v1.5')         
        elif emb_model == "iirc_bge_large_fintune":
            model = SentenceTransformer('/QMKGF-master/embedding_model/large_fintune/iirc_large_fintune')   
    all_start = time.time()
    G = build_graph_from_jsonl(file_path)
    k = 10
    entity_list = get_entities_list(G)
    if datasets == "iirc":
        if list_embedding:
            vector.get_list_vector(embedding, entity_list)
            if emb_model == "bge_large":
                vector.entity_persist(path='/QMKGF-master/vec_store/iirc/iirc_bge_large_entity_dev_list')
            elif emb_model == "iirc_bge_large_fintune":
                vector.entity_persist(path='/QMKGF-master/vec_store/iirc/iirc_bge_large_fintune_entity_dev_list')   
        else:
            if emb_model == "bge_large":
                vector.load_entity_vector('/QMKGF-master/vec_store/iirc/iirc_bge_large_entity_dev_list')
            elif emb_model == "iirc_bge_large_fintune":
                vector.load_entity_vector('/QMKGF-master/vec_store/iirc/iirc_bge_large_fintune_entity_dev_list')
    for question in tqdm(question_list):
        query.append(question)
        ans = answer_list[ans_num]
        ans_num += 1
        extract_time_start = time.time()
        q_entity = entity_extract(question, lan)
        extract_time_end = time.time()
        print(extract_time_end-extract_time_start)
        time1_start = time.time()
        q_entity = entity_map(q_entity, entity_list, embedding, vector)
        time1_end = time.time()
        time1 = time1 + (time1_end - time1_start)
        kg, kg1, kg2, time2_temp, time3_temp = create_kg(question, G, model, q_entity, k, lan)
        time2 += time2_temp
        time3 += time3_temp
        kg = kg_fusion(kg, kg1, kg2, question, embedding)
        all_content = vector.query(question, EmbeddingModel=embedding, k=10, returnall=1)
        content = vector.query(question, EmbeddingModel=embedding, k=10)
        kg_num, kg_content, kg_ref = kg_retrieve_chat(ans, vector, embedding, kg, question, reranker, lan, content, judge_tf, k=3)
        kg_content = protect_empty_string(kg_content)
        pred_answer_kg.append(kg_content)
        pred_answer_kg_num.append(kg_num)
        pred_answer_kg_ref.append(kg_ref)
        if test_num % 50 == 0 and test_num > 0:
            print(f'*******************************{test_num}**********************************')
            print("---------kg-------------------------------------------------------------")
            model_eval(answer_list[:test_num+1], pred_answer_kg, pred_answer_kg_num, lan, judge_tf)
        test_num += 1
    print("---------kg--------------------------------------------------------------")
    model_eval(answer_list, pred_answer_kg, pred_answer_kg_num, lan, judge_tf)
    return 

def dataload(datasets, itor):
    if datasets == "iirc":
        question_list = []
        positives = []
        answer_list = []
        num = 0
        with open("/QMKGF-master/dataset/iirc/dev.jsonl", 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                query = data.get('query')
                answer = data.get('answer')
                if query is not None:
                    question_list.append(query)
                if answer is not None:
                    answer_list.append(str(answer))
                num += 1
                if num == itor:
                    break
            print(len(question_list))
        return question_list, answer_list

def main():
    random.seed(42)
    embedding = BgeEmbedding(path=path)
    reranker = BgeReranker()
    if have_created_db:
        vector = VectorStore()
        if datasets == "iirc":
            if emb_model == "bge_large":
                vector.load_vector('/QMKGF-master/vec_store/iirc/iirc_bge_large') 
            elif emb_model == "iirc_bge_large_fintune":
                vector.load_vector('/QMKGF-master/vec_store/iirc/iirc_bge_large_fintune') 
    else:
        if datasets == "iirc":
            docs = ReadFiles('/QMKGF-master/dataset/iirc').get_jsonl_context()
            vector = VectorStore(docs)
            vector.get_vector(EmbeddingModel=embedding)
            if emb_model == "bge_large":
                vector.persist(path='/QMKGF-master/vec_store/iirc/iirc_bge_large')
            elif emb_model == "iirc_bge_large_fintune":
                vector.persist(path='/QMKGF-master/vec_store/iirc/iirc_bge_large_fintune')
    answer_list, question_list = [], []
    if datasets == "iirc":
        lan = 'english'
        question_list, answer_list = dataload(datasets, itor)
        all_RAG(datasets, embedding, vector, reranker, question_list, answer_list, lan, judge_tf)

main()
print("-------------------end----------------------")
