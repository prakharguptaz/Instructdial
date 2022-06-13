from collections import namedtuple
from datasets import VUPDataset, NUPDataset, MLMDataset
import numpy as np
from data_utils import read_dataset
from models.VUPScorer import VUPScorer
from models.NUPScorer import NUPScorer
from models.MLMScorer import MLMScorer

import argparse
import json

from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_minmax(model, X_data):

    scores = []
    with torch.no_grad():
        for x in tqdm(X_data):
            score = model.predict(x)
            scores.append(score)

    score_dict = {}
    keys = scores[0].keys()
    for k in keys:
        arr = []
        for score in scores:
            arr.append(score[k])        # score of each metric

        # min_s = min(arr)
        # max_s = max(arr)
        min_s = np.quantile(arr, 0.25).item()
        max_s = np.quantile(arr, 0.75).item()

        score_dict[k] = {
            'min': min_s,
            'max': max_s
        }

    return score_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculating min and max of MLM for normalizatiion')
    parser.add_argument('--weight-path', type=str, default='./checkpoints', help='Path to directory that stores the weight')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the directory of training set')
    parser.add_argument('--output-path', type=str, default='mlm_minmax_score.json', help='Output path for the min max values')

    args = parser.parse_args()
    xdata = read_dataset(args.data_path)

    model = MLMScorer.load_from_checkpoint(checkpoint_path=args.weight_path).to(device)
    model.eval()
    print ('[!] loading model complete')

    scores = calc_minmax(model, xdata)
    print ('[!] normalizing complete')

    with open(args.output_path, 'w') as f:
        f.write(json.dumps(scores, indent=4))
        f.close()
    print ('[!] complete')
