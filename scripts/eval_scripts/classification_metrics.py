import argparse
import json
import logging
import numpy as np
import os
import random
import math
import copy
import sklearn.metrics
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict

from scipy import stats
from scipy.stats import pearsonr, spearmanr

from tqdm import tqdm
from tqdm import trange

def correlation_metrics(output, score):
    r_spearmanr, p_spearmanr = spearmanr(output, score)
    r_pearsonr, p_pearsonr = pearsonr(output, score)

    spearmanr_res = str(np.round(r_spearmanr, 3)) + ' (' + str(np.round(p_spearmanr, 3)) + ')'
    pearsonr_res = str(np.round(r_pearsonr, 3)) + ' (' + str(np.round(p_pearsonr, 3)) + ')'
    return  [spearmanr_res, pearsonr_res]

def evaluate_slotrestaurant8k(pred, true, slot_types=["time", "people", "first_name", "last_name", "date"]):
    slot_type_f1_scores = []

    for slot_type in slot_types:
        predictions_for_slot = [
            [p[1] for p in prediction.items() if p[0] == slot_type] for prediction in pred
        ]
        labels_for_slot = [
            [l[1] for l in label.items() if l[0] == slot_type] for label in true
        ]

        proposal_made = [len(p) > 0 for p in predictions_for_slot]
        has_label = [len(l) > 0 for l in labels_for_slot]
        prediction_correct = [
            prediction == label for prediction, label in zip(predictions_for_slot, labels_for_slot)
        ]
        true_positives = sum([
            int(proposed and correct)
            for proposed, correct in zip(proposal_made, prediction_correct)
        ])
        num_predicted = sum([int(proposed) for proposed in proposal_made])
        num_to_recall = sum([int(hl) for hl in has_label])

        precision = true_positives / (1e-5 + num_predicted)
        recall = true_positives / (1e-5 + num_to_recall)

        f1_score = 2 * precision * recall / (1e-5 + precision + recall)
        slot_type_f1_scores.append(f1_score)

        print(slot_type, 'true_positives', true_positives, 'num_predicted', num_predicted, 'num_to_recall', num_to_recall)
        print(slot_type, precision, recall, f1_score)
        # import pdb;pdb.set_trace()
    slot_type_f1_scores = [x for x in slot_type_f1_scores if x!=0]
    return np.mean(slot_type_f1_scores)

# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted, average=None):
  correct = 0 
  for t,p in zip(actual, predicted):
    if t == p or (type(p) is list and t in p):
      correct += 1

  return correct/len(actual)

def auc(actual, predicted):
  assert len(actual) == len(predicted)
  fpr, tpr, thresholds = sklearn.metrics.roc_curve(actual, predicted)
  return sklearn.metrics.auc(fpr, tpr)

def mrr(actual, predicted):
  # Assume actual is sorted in the appropriate order
  mrr = 0
  for t,p in zip(actual, predicted):
    assert t in p
    mrr += (p.index(t) + 1)/len(p)

  return mrr/len(actual)

def correlation(actual, predicted):
  return stats.spearmanr(actual, predicted).correlation
    

# def read_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--configfile", default='configs/taskfiles_config.json', type=str)
#     return parser.parse_args()




def test():
    print('empty')

if __name__ == "__main__":
    test()

