import argparse
import json
import logging
import numpy as np
import os
import random
import math
import copy
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict
from typing import List

from tqdm import tqdm
from tqdm import trange
# pip install bert_score
from nlgeval import NLGEval
import tensorflow as tf
from datasets import load_metric
import sys
from collections import Counter
import argparse
from nltk import ngrams
import re
import nltk



rouge_metric = load_metric('rouge')
bleu_metric = load_metric('bleu')
bertscore_metric = load_metric('bertscore')

tf.compat.v1.flags.DEFINE_string('data','','') # magic to solve bleurt error

def nlgeval_metrics(refs: List[str] , hyps: List[str]):
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=['METEOR'])
    # metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L']
    metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L']
    results = defaultdict(list)
    for ref, hyp in zip(refs, hyps):
        if type(ref) is not list:
            print('PUT REFS in LIST')
            exit(0)
        metrics_dict = nlgeval.compute_individual_metrics(ref, hyp)
        for metric in metrics:
            results[metric].append(metrics_dict[metric])
    return results


def calc_hfbleu_scores(references, candidates):
    refs = []
    cands = []
    hfbleu_scores = []
    for i in range(len(candidates)):
        cand_i = candidates[i]
        if type(cand_i) is not str:
            print('candidate should be string')
            exit(0)
        cand = cand_i.split()
        cands.append(cand)
        ref_i = references[i]
        if type(ref_i) is not list:
            print('refs should be put in list')
            exit(0)
        else:
            ref = []
            for r in ref_i:
                ref.append(r.split())
            refs.append(ref)
        try:
            hfbscore = bleu_metric.compute(predictions=[cand],references=[ref])
            hfbleu_scores.append(hfbscore['bleu']*100)
        except:
            print('weird', cand, ref)
            hfbleu_scores.append(0)

    bleu_scores = bleu_metric.compute(predictions=cands,references=refs)
    
    return {'hfbleu_all':bleu_scores['bleu']*100, 'hfbleu' : hfbleu_scores}

def calc_rouge_scores(references, candidates):
    result = rouge_metric.compute(predictions=candidates, references=references, use_stemmer=True)
    result = {key: round(value.mid.fmeasure * 100, 1) for key, value in result.items()}
    return result

def bert_score(refs: List[str], hyps: List[str]):
    from bert_score import BERTScorer
    scorer = BERTScorer(lang='en', rescale_with_baseline=True)
    P, R, F1 = scorer.score(hyps, refs)
    return {'BERTScore': F1.tolist()}

def bleurt(refs: List[str], hyps: List[str]):
    from bleurt import bscore

    print(os.getcwd())
    # wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
    checkpoint = "eval_scripts/bleurt/bleurt/BLEURT-20"
    scorer = bscore.BleurtScorer(checkpoint)
    if len(refs)>0 and type(refs[0]) is list:
        refs = [x[0] for x in refs]
    scores = scorer.score(references=refs, candidates=hyps)
    return {'BLEURT': scores}

def usl_score(contexts: List[str], hyps: List[str]):

    with open('usl_score/datasets/contexts.txt', 'w') as f:
        for context in contexts:
            f.write(context + '\n')
    with open('usl_score/datasets/hyps.txt', 'w') as f:
        for hyp in hyps:
            f.write(hyp + '\n')
    
    os.system('bash eval_usl.sh')

    scores = []
    with open('usl_score/datasets/score.json') as f:
        for line in f.readlines():
            result = json.loads(line)
            scores.append(result['USL-HS'])
    return {'USL-H': scores}
    

def begins_with_metric(y_pred, y_true, phrases):
    scores = []
    if len(y_pred)==0: return 0
    assert len(phrases) == len(y_pred)
    for i in range(len(y_pred)):
        phrase = phrases[i].lower()
        out = y_pred[i].lower()
        if out.startswith(phrase):
            scores+=[1]
        else:
            scores+=[0]

    return sum(scores)/len(phrases)

def response_length_metric(y_pred, y_true):
    scores = []
    for i in range(len(y_pred)):
        inp = y_true[i].lower()
        out = y_pred[i].lower()
        if len(inp.split())==len(out.split()):
            scores+=[1]
        else:
            scores+=[0]

    return sum(scores)/len(y_pred)

def ends_with_metric(y_pred, y_true, phrases):
    scores = []
    if len(y_pred)==0: return 0
    assert len(phrases) == len(y_pred)
    for i in range(len(y_pred)):
        phrase = phrases[i].lower()
        out = y_pred[i].lower()
        if out.endswith(phrase):
            scores+=[1]
        else:
            scores+=[0]

    return sum(scores)/len(phrases)
# def read_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--configfile", default='configs/taskfiles_config.json', type=str)
#     return parser.parse_args()



re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)

    num_same = sum(common.values())

    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _f1_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for _,_,f1 in scores),max(pre for pre,_,_ in scores),max(rec for _,rec,_ in scores)


def f_one(hypothesis, references):
    '''
    from https://github.com/Nealcly/KE-Blender/blob/main/eval/metrics.py
    calculate f1 metric
    :param hypothesis: list of str
    :param references: list of str
    :return:
    '''
    f1 = []
    pre = []
    rec = []
    for hyp, ref in zip(hypothesis, references):
        res = _f1_score(hyp, [ref])
        f1.append(res[0])
        pre.append(res[1])
        rec.append(res[2])
    return np.mean(f1),np.mean(pre),np.mean(rec)

def knowledge_metric(responses, knowledges):
    '''
    calculate knowledge metric
    :param responses: list of str
    :param knowledges: list of list of str
    :return:
    '''
    stop_words = get_stop_words('en')
    p_scores,  r_scores, f_scores = [], [], []
    for hyp, know in zip(responses, knowledges):
        # hyp_tokens = set([w for w in hyp.split() if w not in stop_words])
        # know = ' '.join(know)
        # know_tokens = set([w for w in know.split() if w not in stop_words])
        #
        # if len(hyp_tokens & know_tokens) == 0:
        #     _p, _r, _f1 = .0, .0, .0
        # else:
        #     _p = len(hyp_tokens & know_tokens) / len(hyp_tokens)
        #     _r = len(hyp_tokens & know_tokens) / len(know_tokens)
        #     _f1 = 2 * (_p * _r) / (_p + _r)

        # hyp_tokens = list(set([w for w in hyp.split() if w not in stop_words]))
        hyp_tokens = [w for w in hyp.split() if w not in stop_words]
        know = ' '.join(know)
        know_tokens = [w for w in know.split() if w not in stop_words]
        _p, _r, _f1 = _prec_recall_f1_score(hyp_tokens, know_tokens)
        p_scores.append(_p)
        r_scores.append(_r)
        f_scores.append(_f1)

    return np.mean(r_scores), np.mean(p_scores),  np.mean(f_scores)

def knowledge_metric_new(responses, knowledges):
    '''
    calculate knowledge metric
    :param responses: list of str
    :param knowledges: list of list of str
    :return:
    '''
    # stop_words = get_stop_words('en')
    # p_scores,  r_scores, f_scores = [], [], []
    # for hyp, know in zip(responses, knowledges):
    #     hyp_tokens = set([w for w in hyp.split() if w not in stop_words])
    #     know = ' '.join(know)
    #     know_tokens = set([w for w in know.split() if w not in stop_words])
    #
    #     if len(hyp_tokens & know_tokens) == 0:
    #         _p, _r, _f1 = .0, .0, .0
    #     else:
    #         _p = len(hyp_tokens & know_tokens) / len(hyp_tokens)
    #         _r = len(hyp_tokens & know_tokens) / len(know_tokens)
    #         _f1 = 2 * (_p * _r) / (_p + _r)
    #
    #     p_scores.append(_p)
    #     r_scores.append(_r)
    #     f_scores.append(_f1)
    #
    # return np.mean(r_scores), np.mean(p_scores),  np.mean(f_scores)

    stop_words = get_stop_words('en')
    p_scores, r_scores, f_scores = [], [], []
    for hyp, know in zip(responses, knowledges):
        hyp_tokens = set([w for w in hyp.split() if w not in stop_words])
        know = ' '.join(know)
        know_tokens = set([w for w in know.split() if w not in stop_words])

        if len(hyp_tokens & know_tokens) == 0:
            _p, _r, _f1 = .0, .0, .0
        else:
            _p = len(hyp_tokens & know_tokens) / len(hyp_tokens)
            _r = len(hyp_tokens & know_tokens) / len(know_tokens)
            _f1 = 2 * (_p * _r) / (_p + _r)

        p_scores.append(_p)
        r_scores.append(_r)
        f_scores.append(_f1)

    return np.mean(r_scores), np.mean(p_scores), np.mean(f_scores)


def test():
    contexts = ['Hello']
    refs = ['How are you']
    hyps = ['How dare you']
    print(usl_score(contexts, hyps))
    print(nlgeval_metrics(refs, hyps))
    print(bert_score(refs, hyps))
    print(bleurt(refs, hyps))

if __name__ == "__main__":
    test()

