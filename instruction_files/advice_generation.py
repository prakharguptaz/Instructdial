from instruction_files.generator_class import GeneratorBasic
from utils import extraction

import string
import json
import random
from string import Template
import os
from collections import Counter, defaultdict
import settings
from tqdm import tqdm
import re

random.seed(123)

instruction_dict = {
    "id": "advice_generation",
    "Source": [
        "self"
    ],
    "Definitions": [
        "In this task you will be shown a text which contains an issue or concern and you need to generate a response that provides advice to resolve the issue.",
        "In this task you will be given a text that raises a issue. Generate a response that gives advice to solve the issue.",
        "In this task you will be given a text which raises a concern. You need to provide a response which resolves the issue."],
    "Positive Examples": [
        {
            "text": settings.CONTEXT_SEP + " Anyone take mental health days from work?. Do you use vacation, PTO, sick, FMLA? " +
                    settings.EOD_SEP + " " +
                    settings.QUESTION_SEP + " The response is ",
            "output": 'Back at my old job, I used sick leave.',
            "index": 1,
            "split": "train",
            "dataset": "advice"
        }
    ]
}


class Generator(GeneratorBasic):
    def __init__(self, args, taskconfig, data_readers):
        self.idx = 0
        self.args = args
        self.taskconfig = taskconfig
        if 'max_data' in self.taskconfig:
            self.max_data = self.taskconfig['max_data']
        else:
            self.max_data = args.max_data
        if 'context_max_length' in self.taskconfig:
            self.context_max_length = self.taskconfig['context_max_length']
        else:
            self.context_max_length = 3
        self.output_folder = args.tasks_output_folder
        self.out_file = self.taskconfig['task_name']
        self.data_readers = data_readers
        self.examples = []

    def generate_data(self):
        print('number of datareaders:', len(self.data_readers))
        sequences = []
        for d, dataset_reader in enumerate(self.data_readers):
            dataset_reader.idx = 0
            iterator_index = 0
            split = dataset_reader.split
            datapoints = []
            dp = dataset_reader.get_next()
            while dp is not None:
                # print(dp)
                iterator_index += 1
                dp = dataset_reader.get_next()
                # if iterator_index>self.max_data:
                #     break
                if dp and 'index' not in dp:
                    dp['index'] = iterator_index
                if dp:
                    datapoints.append(dp)
                    dp['split'] = split
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data * 5))

            definitions = instruction_dict['Definitions']

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            print(len(datapoints), 'datapoints')

            for dp in tqdm(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')

                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-self.context_max_length:])
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                post_prompts = [settings.QUESTION_SEP + " The response is ",
                                settings.QUESTION_SEP + " Given the context an appropriate response is ",
                                settings.QUESTION_SEP + " A good response which resolves the issue in the text is ",
                                settings.QUESTION_SEP + " A helpful response is ",
                                settings.QUESTION_SEP + " The response to this dialog should be "]

                text = settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP + \
                       " " + random.choice(post_prompts)

                text = re.sub(' +', ' ', text)

                sequences.append({'text': text,
                                  'output': dp['response'],
                                  'index': index,
                                  'split': split,
                                  'metadata': {'context':dp['context']},
                                  'dataset': dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
