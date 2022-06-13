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
    "id": "nontoxic_generation",
    "Source": [
        "self"
    ],
    "Definitions": [        
        "In this task you will read a conversion and generate a response that signals that a previous message was not ok",
        "Using the provided conversation, reply with that provides feedback about a toxic utterance",
        "Your goal is to create a response to a toxic message that provides feedback about the toxicity",
        "For this task, you will need to come up with a response to a toxic message that gives feedback about the toxicity",
        "Your task is to read a conversation with a toxic message and generate a response that gives feedback about the toxicity"
        ],
    "Positive Examples": [
        {

            "input": settings.CONTEXT_SEP + " I have never met an italian person, so I can\'t say one way or the other. " +
                     settings.EOT_SEP + " Another group that is lower like that is the Greeks. They are barely able to say one word after the other. " +
                     settings.EOD_SEP + " I\'ve never met a greek person, either. Do you speak any languages other than english? " +
                     settings.EOD_SEP + " No, Americans should not speak anything beyond English. " +
                     settings.QUESTION_SEP + " Given the conversation, a non toxic response is ",
            "output": " don\'t be a racist, we shall encourage diversity. ",
            "index": 0,
            "split": "train",
            "dataset": "saferdialogues"
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

                output = dp['response']

                post_prompts = [settings.QUESTION_SEP + " A response that steers the conversation towards safer direction is ",
                                settings.QUESTION_SEP + " Given the conversation, a safe response is ",
                                settings.QUESTION_SEP + " A response to the conversation that makes it safer is "]

                text = settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP + \
                       " " + random.choice(post_prompts)

                text = re.sub(' +', ' ', text)

                sequences.append(
                    {'text': text, 'output': output, 'index': index,'metadata':{'context':dp['context']}, 'split': split, 'dataset': dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
