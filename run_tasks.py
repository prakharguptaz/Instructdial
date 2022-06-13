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
from tqdm import tqdm
from tqdm import trange

import settings
from constants import SPECIAL_TOKENS


# from data_utils import dialoglue_reader, wow_reader, eval_reader
from datareaders import get_reader
import importlib

# import sys
# sys.path.append("instruction_files")

from sequentialize import get_sequence
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


def get_instructions(instruction_files):
    instruction_folder = settings.INSTRUCTION_FOLDER

    allinstruction_files = [f for f in listdir(instruction_folder) if isfile(join(instruction_folder, f))]
    allinstruction_files = [x.replace('.py', '') for x in allinstruction_files]

    useful_files = []
    for filename in instruction_files:
        for fname in allinstruction_files:
            if filename == fname and not any(c in fname for c in ['swo', 'swp']):
                useful_files.append(fname)
    print('useful_files for instructions', useful_files)
    instructions_modules = []
    for fname in useful_files:
        # if '.py' not in fname: continue
        # fdata = json.load(open(join(instruction_folder, fname), 'r'))
        modulename = fname.replace('.py', '')#join(instruction_folder, fname)
        # module = importlib.import_module(instruction_files.modulename, package='instruction_files')
        module = importlib.import_module('instruction_files.'+modulename)
        instructions_modules.append(module)

    return instructions_modules


def print_samples(dataset_reader, instructions_all):
    for instruction in instructions_all:
        dataset_reader.idx=0
        dp = dataset_reader.get_next()
        print(instruction)
        while dp is not None:
            print(dp)
            sequences = get_sequence(dataset_reader, dp, instruction)
            print("SEQUENCES:", sequences)
            # import pdb;pdb.set_trace()
            # break

            dp = dataset_reader.get_next()

def print_examples(dataset_reader):
    dataset_reader.idx=0
    dp = dataset_reader.get_next()
    iterator_index = 0
    while dp is not None:
        print(dp)
        iterator_index+=1
        dp = dataset_reader.get_next()
        if iterator_index>10:
            break


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--configfile", default='configs/config_task1.json', type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--tasks_output_folder", type=str, default='tasks_files1k')
    parser.add_argument("--max_data", type=int, default=1000000000)
    parser.add_argument("--data_sample_type", type=str, default='common')
    ##common: samples across datareaders, individual: sample for each datareader, max:use all datapoints
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()




def read_examples(args):
    config = json.load(open(args.configfile, 'r'))
    LOGGER.info(config)
        # Data readers 
    # config['datasets'] = ['eval']
    for dataset in config['datasets']:
        if dataset not in config: continue
        datasetconfig = config[dataset]
        instruction_files = datasetconfig['instruction_files']
        if dataset=='intent-clinc':
            token_vocab_name = os.path.basename(datasetconfig['token_vocab_path']).replace(".txt", "")
            dataset_reader = dialoglue_reader.IntentDataset(settings.DIALOGUE_PATH+datasetconfig['train_data_path'],
                                        datasetconfig['max_seq_length'],
                                        token_vocab_name)
        if dataset=='slot-restaurant8k':
            token_vocab_name = os.path.basename(datasetconfig['token_vocab_path']).replace(".txt", "")
            dataset_reader = dialoglue_reader.SlotDataset(settings.DIALOGUE_PATH+datasetconfig['train_data_path'],
                                        datasetconfig['max_seq_length'],
                                        token_vocab_name)
        if dataset=='wow':
            dataset_reader = wow_reader.WoWDataset(settings.WOW_PATH+datasetconfig['train_data_path'],
                                        datasetconfig['max_seq_length'])
        if dataset == 'eval':
            dataset_reader = eval_reader.EvalDataset(settings.EVAL_PATH)
        instructions_all = get_instructions(instruction_files)

        print_examples(dataset_reader)
        # instructions_all = get_instructions(instruction_files)            
        # print_samples(dataset_reader, instructions_all)

def test_instructions(args):
    # Data readers
    config = json.load(open(args.configfile, 'r'))
    task = args.task
    isExist = os.path.exists(args.tasks_output_folder)
    if not isExist:
        os.makedirs(args.tasks_output_folder)
        print("The new output directory is created!")   
    taskconfig = config.get(task, None)
    if taskconfig is not None:
        instruction_files = taskconfig.get('instruction_files', [])
        datasets = taskconfig.get('datasets', [])
        taskconfig['task_name'] = task

    else:
        print('Task does not exist!')
        instruction_files = []

    instructions_all = get_instructions(instruction_files)
    # import pdb;pdb.set_trace()
    for i, instructionmodule in enumerate(instructions_all):
        data_readers = []
        for d, dataset in enumerate(datasets):
            dataset_reader = get_reader(args, dataset)
            dataset_reader.name = dataset
            data_readers.append(dataset_reader)

        # import pdb;pdb.set_trace()
        generator = instructionmodule.Generator(args, taskconfig, data_readers)
        instance_and_instructions = generator.generate_data()

        print("------------Writing--------------")
        generator.sample_and_write(instance_and_instructions)
        # print_examples(dataset_reader)




if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)

    test_instructions(args)

    # to be used for testing tasks
    # read_examples(args)
