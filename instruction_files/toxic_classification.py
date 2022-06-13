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
from utils.common import get_options_string, get_alphabetwithoptions_string

random.seed(123)

instruction_dict = {
    "id": "toxic_classification",
    "Source": [
        "self"
    ],
    "Definitions": [
        "In this task you will be shown either a context and a response you need to determine if the given response is toxic.",
        "Decide if the response is toxic",
        "Look at the conversation context and decided if the provided response is toxic"],
    "Positive Examples": [
        {
            "text": settings.CONTEXT_SEP + " hello! " +
                    settings.EOT_SEP + " hello " +
                    settings.EOT_SEP + " how are you? " +
                    settings.EOT_SEP + " good and you? " +
                    settings.EOD_SEP + " " +
                    settings.RESPONSE_SEP + " not interesting, try again " +
                    settings.QUESTION_SEP + " Is the response toxic? Answer choices [OPTIONS] yes||||no ",
            "output": "Yes",
            "index": 10734,
            "split": "train",
            "dataset": "toxichat"
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
                toxic = dp['toxic']
                toxic = 'yes' if toxic else 'no'
                post_prompts = [settings.QUESTION_SEP + " Is the response toxic? ",
                                settings.QUESTION_SEP + " Given the conversation, is the provided response toxic? "]

                text = settings.CONTEXT_SEP + " " + context_str  + " " + \
                       settings.RESPONSE_SEP + " " + dp['response'] + " " + settings.EOD_SEP + \
                       " " + random.choice(post_prompts) + " " + get_options_string(["yes", "no"])

                sequences.append({'input': text,
                                  'output': toxic,
                                  'index': index,
                                  'split': split,
                                  'dataset': dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
