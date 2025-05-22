# coding=utf-8

import json
import warnings
import numpy as np
# import nltk
# from typing import List
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk import word_tokenize
from meteor import Meteor
import re
split_punct = lambda x:" ".join(y for y in re.findall(r"[\w']+|[.,!?;]", x))

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {file_path}")

def extract_response(response):
    answer_pattern = r"<response>(.*?)</response>"
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    answer = ""
    if answer_match:
        answer = answer_match.group(1).strip()
    return answer

def get_refs_hyps(data):
    refs = []
    hyps = []
    for item in data:
        ref = item["gold_response"]
        hyp = item["response"]
        refs.append([ref])
        hyps.append(hyp)
    return refs, hyps

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for _ in range(0,len(sub)+1)] for _ in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]


class Metric(object):
    def __init__(self, refs, hyps):
        self.origin_refs = refs
        self.origin_hyps = hyps
        self.refs = []
        self.hyps = []
        self.meteor = Meteor()
        for i in range(len(hyps)):
            self.forword(refs[i], hyps[i])
    def forword(self, refs: str, hyp: str):
        self.refs.append([word_tokenize(split_punct(e).lower()) for e in refs])
        self.hyps.append(word_tokenize(split_punct(hyp.lower()))) 
    def calc_bleu_k(self, k):
        weights = [1. / k] * k + (4 - k) * [0.]
        try:
            bleu = corpus_bleu(self.refs, self.hyps, weights=weights, smoothing_function=SmoothingFunction().method3)
        except ZeroDivisionError as _:
            warnings.warn('the bleu is invalid')
            bleu = 0.
        return bleu

    def calc_distinct_k(self, k):
        d = {}
        tot = 0
        for sen in self.hyps:
            for i in range(0, len(sen)-k):
                key = tuple(sen[i:i+k])
                d[key] = 1
                tot += 1
        if tot > 0:
            dist = len(d) / tot
        else:
            warnings.warn('the distinct is invalid')
            dist = 0.
        return dist
    
    def calc_unigram_f1(self):
        f1_scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            scores = []
            for ref in refs:
                cross = Counter(hyp) & Counter(ref)
                cross = sum(cross.values())
                p = cross / max(len(hyp), 1e-10)
                r = cross / max(len(ref), 1e-10)
                f1 = 2 * p * r / max(p + r, 1e-10)
                scores.append(f1)
            f1_scores.append(max(scores))
        return np.mean(f1_scores), f1_scores
    
    def calc_rouge_l(self, beta=1.2):
        scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            prec = []
            rec = []
            for ref in refs:
                lcs = my_lcs(ref, hyp)
                prec.append(lcs / max(len(hyp), 1e-10))
                rec.append(lcs / max(len(ref), 1e-10))
            prec_max = max(prec)
            rec_max = max(rec)
            if prec_max != 0 and rec_max !=0:
                score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + beta**2*prec_max)
            else:
                score = 0.0
            scores.append(score)
        return np.mean(scores), scores
        
    def calc_meteor(self):
        gts = {}
        res = {}
        idx = 0
        for hyp, refs in zip(self.origin_hyps, self.origin_refs):
            gts[str(idx)] = refs
            res[str(idx)] = [hyp]
            idx+=1
        # print(gts)
        # print("="*20)
        # print(res)
        score, _ = self.meteor.compute_score(gts, res)
        return score

    def close(self):
        result = {
            'length': float(np.mean(list(map(len, self.hyps)))),
            **{f"dist-{k}": 100 * self.calc_distinct_k(k) for k in range(1, 4)},
            **{f"bleu-{k}": 100 * self.calc_bleu_k(k) for k in range(1, 5)}
        }
        
        f1, scores = self.calc_unigram_f1()
        result['f1'] = 100 * f1
        result_list = {
            'f1': scores
        }
        
        rl, scores = self.calc_rouge_l()
        result['rouge-l'] = 100 * rl
        result_list.update({
            'rouge-l': scores
        })
        
        result['meteor'] = 100 * self.calc_meteor()
        return result, result_list


file_paths = ["llama_inference.json"]

input_file_dir = "./test_results"
output_file_dir = "./test_metrics"
for file_path in file_paths:
    input_file_name = f"{input_file_dir}/{file_path}"
    output_file_name = f"{output_file_dir}/{file_path}"
    data = load_data(input_file_name)
    refs, hyps = get_refs_hyps(data)
    metric = Metric(refs, hyps)
    result, _ = metric.close()
    for key,val in result.items():
        result[key] = round(val, 2)
    print(result)
    save_data(output_file_name, result)