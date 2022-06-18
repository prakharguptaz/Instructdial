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


class DNLIDataset(Dataset):
    def __init__(self,split='train'):
        self.split = split
        # For caching
        # data_dirname = os.path.dirname(os.path.abspath(data_path))
        # split = os.path.basename(os.path.abspath(data_path))
        self.examples = []
        self.labels = ['positive', 'negative', 'neutral']
        self.name = 'dnli'

        if split == 'train':
            data = json.load(open('./datasets/dnli/dialogue_nli/dialogue_nli_train.jsonl'))
            self.examples+=data
            # data = json.load(open('./datasets/dialogue_nli/dnli/dialogue_nli_extra/repaired_train.json'))
            # self.examples+=data
        if split == 'dev':
            data = json.load(open('./datasets/dnli/dialogue_nli/dialogue_nli_dev.jsonl'))
            self.examples+=data
            # data = json.load(open('./datasets/dialogue_nli/dnli/dialogue_nli_extra/dialogue_nli_EXTRA_uu_test.jsonl'))
            # self.examples+=data
        if split == 'test':
            data = json.load(open('./datasets/dnli/dialogue_nli/dialogue_nli_verified_test.jsonl'))
            self.examples+=data
            # data = json.load(open('./datasets/dialogue_nli/dnli/dialogue_nli_extra/dialogue_nli_EXTRA_uu_dev.jsonl'))
            # self.examples+=data

        print('data length', len(self.examples))

        # for i, dialogue in enumerate(lines):
        #     self.examples.append(dialogue)

        self.idx=0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


