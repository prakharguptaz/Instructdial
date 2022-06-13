import csv
import logging
import json
import numpy as np
import os
import pickle
import string
import random
from collections import defaultdict
from tokenizers import BertWordPieceTokenizer
# from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict
import jsonlines

from constants import SPECIAL_TOKENS

from data_utils.data_reader import Dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

def get_json_lines(inp_file):
    lines = []
    with jsonlines.open(inp_file) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


class DECODEDataset(Dataset):
    def __init__(self,split='train'):
        self.split = split
        # For caching
        # data_dirname = os.path.dirname(os.path.abspath(data_path))
        # split = os.path.basename(os.path.abspath(data_path))
        self.examples = []
        self.labels = ['contradicted', 'uncontradicted']
        self.name = 'dnli'

        if split == 'train':
            data = get_json_lines(('./datasets/decode_v0.1/train.jsonl'))
        if split == 'dev':
            data = get_json_lines(('./datasets/decode_v0.1/dev.jsonl'))
        if split == 'test':
            data = get_json_lines(('./datasets/decode_v0.1/test.jsonl'))

        for i, odp in enumerate(tqdm(data)):
            # print(dp)
            # import pdb;pdb.set_trace()
            if len(odp['aggregated_contradiction_indices'])>0:
                dialogue = [t['text'] for t in odp['turns']]
                dp = dict()
                dp['context'] = dialogue[:-1]
                dp['response'] = dialogue[-1]
                dp['label'] = 'contradicted'
                self.examples.append(dp)

                num_trials = 0
                randomindex = random.choice([x for x in range(len(dialogue))])
                while randomindex in odp['aggregated_contradiction_indices']:
                    randomindex = random.choice([x for x in range(1, len(dialogue))])
                    num_trials += 1
                    if num_trials>10:
                        num_trials = -1
                        break
                if num_trials!=-1:
                    dp = dict()
                    dp['context'] = dialogue[:randomindex]
                    dp['response'] = dialogue[randomindex]
                    dp['label'] = 'uncontradicted'
                    self.examples.append(dp)



        print('data length', len(self.examples))

        # for i, dialogue in enumerate(lines):
        #     self.examples.append(dialogue)

        self.idx=0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


