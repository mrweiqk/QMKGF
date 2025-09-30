from rouge import Rouge
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import numpy as np
import re
from collections import Counter


class eval_test():
    def __init__(self) -> None:
        self.bleu_1 = 0
        self.bleu_2 = 0
        self.bleu_3 = 0
        self.bleu_4 = 0
        self.rouge_1 = 0
        self.rouge_2 = 0
        self.rouge_l = 0
        self.meteor = 0
        pass

    def preprocess_text(self,text):
        words = jieba.lcut(text)
        return ' '.join(words)

    def preprocess_text_meteor(self,text):
        words = jieba.lcut(text)
        return words

    def preprocess_text_english(self,text):
        words = re.findall(r'\b\w+\b', text)
        return ' '.join(words)

    def replace_empty_or_none(self,s):
        if s is None or not isinstance(s, str) or s.strip() == "":
            return "Empty"
        if s.strip() == '[]':
            return "Empty"
        return [s if item == '[]' else item for item in s]

    
    def calculate_rouge_l(self,reference, candidate,lan):
        rouge = Rouge()

        if lan == 'chinese':
            reference = self.preprocess_text(reference)
            candidate = self.preprocess_text(candidate)
        elif lan == 'english':
            reference = self.preprocess_text_english(reference)
            candidate = self.preprocess_text_english(candidate)

        scores = rouge.get_scores(candidate, reference)
        return scores[0]['rouge-l']['f'], scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f']
    def rouge_eval(self,answer_list, perd_answer_list,lan):
        rouge_l_scores, rouge_1_scores, rouge_2_scores = [],[],[]
        total_rouge_1_score,total_rouge_2_score,total_rouge_l_score = 0,0,0
        length = len(answer_list)
        for i in range(length):
            ans = answer_list[i].replace('\n', '')
            pred = perd_answer_list[i].replace('\n', '')


            rouge_l, rouge_1, rouge_2 = self.calculate_rouge_l(ans, pred,lan)
            rouge_l_scores.append(rouge_l)
            rouge_1_scores.append(rouge_1)
            rouge_2_scores.append(rouge_2)

        for num in rouge_1_scores:
            total_rouge_1_score += num    
        print("rouge_1_scores = ", total_rouge_1_score / len(rouge_l_scores))
        for num in rouge_2_scores:
            total_rouge_2_score += num    
        print("rouge_2_scores = ", total_rouge_2_score / len(rouge_l_scores))
        for num in rouge_l_scores:
            total_rouge_l_score += num   
        print("rouge_l_scores = ", total_rouge_l_score / len(rouge_l_scores))
        self.rouge_l = total_rouge_l_score / len(rouge_l_scores)
        self.rouge_1 = total_rouge_1_score / len(rouge_l_scores)
        self.rouge_2 = total_rouge_2_score / len(rouge_l_scores)


    def calculate_blue_eval(self,answer_list,perd_answer_list,lan):
        if lan == 'chinese':
            reference_tokens = self.preprocess_text(answer_list)
            candidate_tokens = self.preprocess_text(perd_answer_list)
        if lan == 'english':
            reference_tokens = self.preprocess_text_english(answer_list)
            candidate_tokens = self.preprocess_text_english(perd_answer_list)

        smoothing_function = SmoothingFunction().method1

        bleu1=sentence_bleu([reference_tokens], candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
        bleu2 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
        bleu3 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
        bleu4 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
        return bleu1,bleu2,bleu3,bleu4

    def bleu_eval(self,answer_list, perd_answer_list,lan):
        bleu_1_scores,bleu_2_scores,bleu_3_scores, bleu_4_scores= [],[],[],[]
        total_bleu_1,total_bleu_2,total_bleu_3,total_bleu_4 = 0,0,0,0
        length = len(answer_list)
        for i in range(length):
            ans = answer_list[i].replace('\n', '')
            pred = perd_answer_list[i].replace('\n', '')
            bleu_1, bleu_2, bleu_3, bleu_4 = self.calculate_blue_eval(ans, pred,lan)
            bleu_1_scores.append(bleu_1)
            bleu_2_scores.append(bleu_2)
            bleu_3_scores.append(bleu_3)
            bleu_4_scores.append(bleu_4)

        for num in bleu_1_scores:
            total_bleu_1 += num   
        print("total_bleu_1 = ", total_bleu_1 / len(bleu_1_scores))
        for num in bleu_2_scores:
            total_bleu_2 += num   
        print("total_bleu_2 = ", total_bleu_2 / len(bleu_1_scores))
        for num in bleu_3_scores:
            total_bleu_3 += num   
        print("total_bleu_3 = ", total_bleu_3 / len(bleu_1_scores))
        for num in bleu_4_scores:
            total_bleu_4 += num   
        print("total_bleu_4 = ", total_bleu_4 / len(bleu_1_scores))
        self.bleu_1 = total_bleu_1 / len(bleu_1_scores)
        self.bleu_2 = total_bleu_2 / len(bleu_1_scores)
        self.bleu_3 = total_bleu_3 / len(bleu_1_scores)
        self.bleu_4 = total_bleu_4 / len(bleu_1_scores)


    def calculate_meteor(self,answer_list, perd_answer_list,lan):
                    
        reference = self.preprocess_text_meteor(answer_list)
        perd_answer_list = perd_answer_list.replace('\n', '')
        candidate = self.preprocess_text_meteor(perd_answer_list)

        score = meteor_score([reference], candidate)
        return score

    def meteor_eval(self,answer_list, perd_answer_list,lan):
        meteor_scores = []
        total_meteor = 0
        length = len(answer_list)
        for i in range(length):
            ans = answer_list[i]
            pred = perd_answer_list[i]
            meteor = self.calculate_meteor(ans, pred,lan)
            meteor_scores.append(meteor)
        for num in meteor_scores:
            total_meteor += num   
        print("total_meteor = ", total_meteor / len(meteor_scores))
        self.meteor = total_meteor / len(meteor_scores)

    def save_to_txt(self, filename="evaluator_data.txt"):  

        with open(filename, "w", encoding="utf-8") as file:  

            file.write(f"BLEU_1 score: {self.bleu_1}\n") 
            file.write(f"BLEU_2 score: {self.bleu_2}\n") 
            file.write(f"BLEU_3 score: {self.bleu_3}\n") 
            file.write(f"BLEU_4 score: {self.bleu_4}\n") 
            file.write(f"Rouge_1 score: {self.rouge_1}\n") 
            file.write(f"Rouge_2 score: {self.rouge_2}\n") 
            file.write(f"Rouge_l score: {self.rouge_l}\n") 
            file.write(f"Meteor score: {self.meteor}\n")  
