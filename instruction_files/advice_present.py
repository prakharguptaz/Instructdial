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
    "id": "advice_present",
    "Source": [
        "self"
    ],
    "Definitions": [
        "In this task you will be shown a text which contains an issue or concern and a response. You need to determine if the response provides advice for the issue in the text",
        "In this task you will be given a text that raises a issue. Decide if the response gives advice to solve the issue.",
        "In this task you will be given a text which raises a concern. You need to decided if the provided response resolves the issue."],
    "Positive Examples": [
        {
            "text": settings.CONTEXT_SEP + " Anyone take mental health days from work?. Do you use vacation, PTO, sick, FMLA? " +
                    settings.EOD_SEP + " " +
                    settings.RESPONSE_SEP + " Back at my old job, I used sick leave. " +
                    settings.QUESTION_SEP + " Does the response provide advice for the issue in the text? Answer choices [OPTIONS] yes||||no ",
            "outputs": ['Yes'],
            "explanation": ""
        }
    ],
    "Negative Examples": [
        {
            "text": settings.CONTEXT_SEP + " I don\'t plan on seeing the solar eclipse but I just wanted to know why it\'s dangerous to look at it even though most of the sun is covered. " +
                    settings.EOD_SEP + " " +
                    settings.RESPONSE_SEP + " Back at my old job, I used sick leave. " +
                    settings.QUESTION_SEP + " Does the response provide advice for the issue in the text? Answer choices [OPTIONS] yes||||no ",
            "outputs": ["No"],
            "explanation": ""
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

            print(len(datapoints), 'datapoints')

            for dp in tqdm(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')

                post_prompts = [settings.QUESTION_SEP + " Does the response provide advice for the issue in the text? ",
                                settings.QUESTION_SEP + " Is the response appropriate? ",
                                settings.QUESTION_SEP + " Is the response helpful? ",
                                settings.QUESTION_SEP + " Does the response help resolve the issue raised in the text?"]

                advice = random.choice(dp['comments'])
                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-self.context_max_length:])
                context = context + ' ' + settings.RESPONSE_SEP + ' ' + advice
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                text = settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP + \
                       " " + random.choice(post_prompts) + " " + get_options_string(["yes", "no"])

                sequences.append({'input': text,
                                  'outputs': ['yes'],
                                  'index': index,
                                  'split': split,
                                  'metadata': {'context':dp['context'], 'response':advice},
                                  'dataset': dataset_reader.name})

                i = random.randint(0, len(datapoints) - 1)
                while i == index:
                    i = random.randint(0, len(datapoints) - 1)

                random_advice = random.choice(datapoints[i]['comments'])
                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-self.context_max_length:])
                context = context + ' ' + settings.RESPONSE_SEP + ' ' + random_advice
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                text = settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP + \
                       " " + random.choice(post_prompts) + " " + get_options_string(["yes", "no"])

                sequences.append({'input': text,
                                  'outputs': ['no'],
                                  'index': index,
                                  'split': split,
                                  'metadata': {'context':dp['context'], 'response':random_advice},
                                  'dataset': dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
