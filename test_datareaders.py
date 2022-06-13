import argparse
import json
import logging
import numpy as np
import os
import random
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange

import settings
from constants import SPECIAL_TOKENS
from datareaders import get_reader

from data_utils import atis_reader, cqa_reader, dialoglue_reader, dnli_reader, multiwoz_reader
from data_utils import eval_reader, hf_reader, otters_reader, snips_reader, wow_reader, me2e_reader, airdialoglue_reader
from data_utils import taskmaster_reader, deal_reader, casino_reader, empathy_reader

from sequentialize import get_sequence

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


def print_examples(dataset_reader):
    dataset_reader.idx = 0
    dp = dataset_reader.get_next()
    iterator_index = 0
    while dp is not None:
        print(dp)
        iterator_index += 1
        dp = dataset_reader.get_next()
        if iterator_index > 10:
            break


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--configfile", default='configs/config1.json', type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def test_readers(args, dataset):
    # Data readers
    config = json.load(open(args.configfile, 'r'))

    datasetconfig = config.get(dataset, None)
    if datasetconfig is not None:
        instruction_files = datasetconfig.get('instruction_files', [])
    else:
        instruction_files = []

    dataset_reader = get_reader(args, dataset)

    print_examples(dataset_reader)


if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)

    config = json.load(open(args.configfile, 'r'))
    passed = []
    failed = []
    for dataset in config.keys():
        if dataset == "datasets":
            continue

        try:
            print("Testing dataset={}".format(dataset))
            test_readers(args, dataset)
        except Exception as e:
            print(e)
            print("FAILED dataset={}".format(dataset))
            failed.append(dataset)
            continue

        print("PASSED dataset={}".format(dataset))
        passed.append(dataset)

    print("The following datasets + datareaders worked successfully:", ", ".join(passed))
    print("The following datasets + datareaders FAILED:", ", ".join(failed))
