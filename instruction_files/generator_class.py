import string
import json
import random
from string import Template
import os
from collections import Counter, defaultdict
import settings
from tqdm import tqdm


class GeneratorBasic:
    def __init__(self, args, taskconfig, data_readers):
        self.idx=0
        self.args = args
        self.taskconfig = taskconfig
        if 'max_data' in self.taskconfig:
            self.max_data = self.taskconfig['max_data']
        else:
            self.max_data = args.max_data
        self.output_folder = args.tasks_output_folder
        self.out_file = self.taskconfig['task_name']
        self.data_readers = data_readers
        self.examples = []

    def generate_data(self):
        print('generating basic')

    def sample_and_write(self, instance_and_instructions):
        if type(instance_and_instructions) is None:
            print('Please refactor code so that the sample_and_write function performs sampling and writing')
        sequences, instruction_dict = instance_and_instructions
        print('len datareader_sequences Instances', len(sequences))
        if self.args.data_sample_type=='individual':
            data_dict_all = defaultdict(list)
            for dp in sequences:
                # import pdb;pdb.set_trace()
                data_dict_all[dp['dataset']].append(dp)
            # for d, dataset_reader in enumerate(self.data_readers):
            for dataname in data_dict_all.keys():
                datareader_sequences = data_dict_all[dataname]
                random.shuffle(datareader_sequences)
                # print('len(sequences)', len(datareader_sequences))
                datareader_sequences_sampled = random.sample(datareader_sequences, min(len(datareader_sequences), self.max_data))
                if 'Instances' not in instruction_dict: instruction_dict['Instances'] = []
                instruction_dict['Instances'] += datareader_sequences_sampled[:]
                # sequences = []

        elif self.args.data_sample_type=='common':
            random.shuffle(sequences)
            sequences = random.sample(sequences, min(len(sequences), self.max_data))
            instruction_dict['Instances'] = sequences

        else:# self.args.data_sample_type=='max':
            instruction_dict['Instances'] = sequences

        for s in sequences[-5:]:
            print(s)

        print('writing to', self.out_file, 'size of data', len(sequences))

        with open(self.output_folder+'/'+self.out_file+'.json', 'w') as f:
            f.write(json.dumps(instruction_dict, indent=2))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
