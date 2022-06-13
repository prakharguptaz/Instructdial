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


def clean_target_fromresponse(response, target):
    targetnopunc = target.translate(str.maketrans('', '', string.punctuation)).lower()
    response = response.replace(target, '').replace(target.lower(), '').replace(targetnopunc, '').strip()

    return response

class TopicalChatDataset(Dataset):
    def __init__(self,split='train',type_seen='rare'):
        self.split = split
        self.type_seen = type_seen
        # For caching
        # data_dirname = os.path.dirname(os.path.abspath(data_path))
        # split = os.path.basename(os.path.abspath(data_path))
        self.examples = []

        file_name = split
        if 'valid' in split or 'test' in split:
            file_name = split + '_' + type_seen
        else:
            self.type_seen = ''

        folder = './datasets/Response-Generation-Baselines/processed_output/'
        all_src = []
        with open(folder + file_name +'.src') as csv_file:
            # csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_file:
                all_src.append(row.strip())
                line_count+=1
            # print(f'Processed {line_count} lines.')
        all_target = []
        with open(folder + file_name +'.tgt') as csv_file:
            line_count = 0
            for row in csv_file:
                all_target.append(row.strip())
                line_count+=1

        all_facts = []
        with open(folder + file_name +'.fct') as csv_file:
            line_count = 0
            for row in csv_file:
                all_facts.append(row.strip())
                line_count+=1

        # print(len(all_src), len(all_facts), len(all_target))
        assert len(all_src) == len(all_facts) == len(all_target)
            # print(f'Processed {line_count} lines.')

        combined_data = []
        for i in range(len(all_src)):
            source = all_src[i].strip().split(' _eos ')
            if len(source)>0: source[-1] = source[-1].replace(' _eos', '')
            target = all_target[i].replace('_eos', '').replace('_go', '').strip()
            fact = all_facts[i]
            d = dict()
            d['context'] = source
            d['response'] = target
            d['knowledge']= fact
            d['idx'] = i
            d['type_seen'] = self.type_seen
            combined_data.append(d)

        # print('indomain data length', len(combined_data))
        self.examples+=combined_data


        print('data length', len(self.examples))

        # for i, dialogue in enumerate(lines):
        #     self.examples.append(dialogue)

        self.idx=0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


