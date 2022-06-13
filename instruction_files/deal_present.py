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
from utils.common import get_options_string, get_alphabetwithoptions_string

random.seed(123)

instruction_dict = {
    "id": "deal_present",
    "Source": [
        "self"
    ],
    "Definitions": [
        "In this task you will be shown a conversation you have to determine if an agreement was reached ",
        "Read the given conversation and determine if the people in the conversation came to an agreement",
        "In this task you will be shown a conversation where people are negotiating. Determine if the people in the conversation reached an agreement"],
    "Positive Examples": [
        {
            "text": settings.CONTEXT_SEP + " i\'d like the basketball and the hat , you can keep all 4 books ? " +
                    settings.EOT_SEP + " deal " + settings.EOD_SEP + " " +
                    settings.QUESTION_SEP + " Was an agreement reached? Answer choices [OPTIONS] yes||||no ",
            "output": "Yes",
            "index": 17,
            "dataset": "deal"
        }
    ],
    "Negative Examples": [
        {
            "text": settings.CONTEXT_SEP + " you are only wasting your time " +
                    settings.EOT_SEP + " balls and hat to me , book to you " +
                    settings.EOT_SEP + " lol good try .  " + settings.EOD_SEP + " " +
                    settings.QUESTION_SEP + " Is there an agreement in this dialogue? Answer choices [OPTIONS] yes||||no  ",

            "output": "No",
            "index": 1743,
            "split": "train",
            "dataset": "deal"
        }
    ]
}


def list_tostring(classes):
    assert type(classes) == list
    lenc = len(classes)
    if len(classes) < 2:
        return ' '.join(classes)
    elif len(classes) == 2:
        return classes[0] + ' and ' + classes[1]
    else:
        return ', '.join(classes[:-1]) + ' and ' + classes[-1]


class Generator(GeneratorBasic):
    def __init__(self, args, taskconfig, data_readers):
        self.idx = 0
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

                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                if dp['agreement']:
                    output = 'yes'
                else:
                    output = "no"

                post_prompts = [settings.QUESTION_SEP + " Was an agreement reached? ",
                                settings.QUESTION_SEP + " Is there an agreement between the people in this dialogue? ",
                                settings.QUESTION_SEP + " Is the negotiation in dialogue resolved ? ",
                                settings.QUESTION_SEP + " Is an agreement reached between the people negotiating ? "]

                text = settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP \
                       + " " + random.choice(post_prompts) + " " + get_options_string(["yes", "no"])
                text = re.sub(' +', ' ', text)

                sequences.append({'text': text,
                                  'output': output,
                                  'index': index,
                                  'metadata':{ 'context':dp['context']},
                                  'split': split,
                                  'dataset': dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
