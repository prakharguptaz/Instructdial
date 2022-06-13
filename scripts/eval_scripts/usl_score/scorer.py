import os
import json
import argparse
from tqdm.auto import tqdm
import numpy as np
import torch
from collections import namedtuple
from models import VUPScorer, NUPScorer, MLMScorer, distinct, composite_one_instance
from data_utils import encode_truncate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Scorer:

    def __init__(self, args):
        self.args = args
        vup_path = os.path.join(args.weight_dir, 'BERT-VUP.ckpt')
        nup_path = os.path.join(args.weight_dir, 'BERT-NUP.ckpt')
        mlm_path = os.path.join(args.weight_dir, 'BERT-MLM.ckpt')
        self.vup_model = VUPScorer.load_from_checkpoint(checkpoint_path=vup_path).to(device)
        self.nup_model = NUPScorer.load_from_checkpoint(checkpoint_path=nup_path).to(device)
        self.mlm_model = MLMScorer.load_from_checkpoint(checkpoint_path=mlm_path).to(device)

        # load normalize score
        norm_score_path = os.path.join(args.weight_dir, 'mlm_minmax_score.json')
        self.norm_scores = None
        with open(norm_score_path) as f:
            self.norm_scores = json.load(f)
            f.close()

        if self.norm_scores is None:
            print ('[!] missing normalize file. can not run normalize score on MLM')

        print ('[!] loading models comlete')

    def get_scores(self, contexts, responses, normalize=False):
        scores = []
        for c, r in tqdm(zip(contexts, responses)):
            if c.strip() == "" or r.strip() == "":
                continue
            score = self.get_score(c, r, normalize=normalize)
            scores.append(score)

        keys = scores[0].keys()
        avg_scores = {}
        for k in keys:
            arr = []
            for score in scores:
                arr.append(score[k])
            avg = sum(arr) / len(arr)
            avg_scores[k] = avg


        distinct_score = self.get_distinct(responses)
        for k,v in distinct_score.items():
            avg_scores[k] = v

        return avg_scores, scores


    def get_score(self, context, response, normalize=False):
        scores = {}
        vup_score = self.get_vup(response)
        nup_score = self.get_nup(context, response)
        mlm_score = self.get_mlm(response, normalize=True)

        scores['vup'] = vup_score
        scores['nup'] = nup_score
        for k,v in mlm_score.items():
            scores[k] = v

        usl_hs = self.get_composite(vup_score, nup_score, scores['norm_nll'], method='HS')
        scores['USL-HS'] = usl_hs
        return scores


    def get_vup(self, response):
        return self.vup_model.predict(response)

    def get_nup(self, context, response):
        return self.nup_model.predict(context, response)

    def get_mlm(self, response, normalize=False):
        scores = self.mlm_model.predict(response)

        if normalize:
            assert self.norm_scores is not None
            keys = list(scores.keys())
            for k in keys:
                scores[f'norm_{k}'] = self.minmax_normalize(
                                scores[k],
                                self.norm_scores[k]['min'],
                                self.norm_scores[k]['max']
                            )
        return scores

    def get_composite(self, u, s, l, method='HS', coef=[0.33, 0.33, 0.34]):
        return composite_one_instance(u, s, l, method, coef)

    def minmax_normalize(self, score, min_score, max_score):
        score = score if score >= min_score else min_score
        score = score if score <= max_score else max_score
        return (score - min_score) / (max_score - min_score)

    def get_distinct(self, responses):
        scores = distinct(responses)
        return scores
