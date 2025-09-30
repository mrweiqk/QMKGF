# !/usr/bin/env python3

import torch
from rich import print
from transformers import AutoTokenizer
def cal(line1,line2,line3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(
        '/QMKGF-master/reward/rewardtransformers/RLHF/checkpoints/reward_model/kg_reward/model_best',
        local_files_only=True
    )

    model = torch.load(
        '/QMKGF-master/reward/rewardtransformers/RLHF/checkpoints/reward_model/kg_reward/model_best/model.pt',
        weights_only=False
    )

    model.to(device).eval()

    def triples_to_text_with_brackets(triple_list):
        return "".join([f"({h}, {r}, {t})" for h, r, t in triple_list])

    texts = [
        triples_to_text_with_brackets(line1),
        triples_to_text_with_brackets(line2),
        triples_to_text_with_brackets(line3)
    ]

    inputs = tokenizer(
        texts, 
        max_length=128,
        padding='max_length', 
        truncation=True,    
        return_tensors='pt'   
    )
    

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        r = model(**inputs)


    print(r) 
    best_index = torch.argmax(r).item() 
    return best_index


    

def test():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    tokenizer = AutoTokenizer.from_pretrained(
        '/QMKGF-master/reward/rewardtransformers/RLHF/checkpoints/reward_model/kg_reward/model_best',
        local_files_only=True
    )

    # 加载模型并设置 weights_only=False（如果你信任该文件来源）
    model = torch.load(
        '/QMKGF-master/reward/rewardtransformers/RLHF/checkpoints/reward_model/kg_reward/model_best/model.pt',
        weights_only=False
    )

    # 将模型移动到指定设备并设置为评估模式
    model.to(device).eval()
# # 输入文本
    texts = [
        "('三江县仙人山景区', '三江县仙人山景区 提供侗族百家宴，让游客品尝正宗侗族美食。', '侗族百家宴')('三江县仙人山景区', '三江县仙人山景区 提供侗族民俗活动，包括表演和体验。', '侗族民俗活动')",
        "('三江县仙人山景区', '三江县仙人山景区 提供侗族民俗活动，包括表演和体验。', '侗族民俗活动')('三江县仙人山景区', '三江县仙人山景区 提供侗族百家宴，让游客品尝正宗侗族美食。', '侗族百家宴')",
        "('刘冯故居景区', '刘冯故居景区 内的历史传承体现了当地的文化底蕴。', '历史传承')('刘冯故居景区', '刘冯故居景区 让游客有机会深入了解当地的历史和文化。', '当地历史和文化')('刘冯故居景区', '刘冯故居景区 内的建筑风格展示了当地的历史文化。', '建筑风格')",
        "('麻垌荔枝', '麻垌荔枝也可以用来酿制荔枝酒，这是一种独特的风味饮品。', '荔枝酒')('麻垌荔枝', '丁香荔是麻垌荔枝的一种品种。', '丁香荔')('麻垌荔枝', '通过晒干荔枝汁制成的干茶叶可以用来泡制荔枝茶，味道香甜。麻垌荔枝可以用来制作荔枝茶，这是一种香甜的饮品。', '荔枝茶')('麻垌荔枝', '黑叶荔是麻垌荔枝的一种品种。', '黑叶荔')"
    ]

    # 对文本进行编码
    inputs = tokenizer(
        texts, 
        max_length=128,
        padding='max_length',  # 启用填充
        truncation=True,       # 启用截断
        return_tensors='pt'    # 返回 PyTorch 张量
    )
    
    # **关键修改：将输入数据移动到指定设备**
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 模型推理
    with torch.no_grad():  # 禁用梯度计算以节省内存
        r = model(**inputs)

    # 打印结果
    print(r)

def main():
    line1 = []
    line2 = []
    line3 = []
    # cal(line1,line2,line3)
    test()

if __name__ == "__main__":
    main()