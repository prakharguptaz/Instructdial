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
        "In this task you will read a conversion and the task is to generate a non toxic response that takes the conversation to a safer direction based on the feedback from the last response",
        "Using the provided conversation, reply with a response that is not toxic and takes the conversation on a more acceptable and friendly trajectory",
        "Provide a response to the dialog in a polite and considerate way that acknowledges the feedback, is not argumentative, and takes the conversation on a more acceptable and friendly trajectory",
        "In this task you will read a conversion and generate a recovery response to a feedback about a toxic response",
        "Using the provided conversation, reply with a response that is a recovery response to the toxic text in the conversation",
        "Provide a response to the dialog that recovers from previously mentioned toxic text"],
    "Positive Examples": [
        {

            "input": settings.CONTEXT_SEP + " I have never met an italian person, so I can\'t say one way or the other. " +
                     settings.EOT_SEP + " Another group that is lower like that is the Greeks. They are barely able to say one word after the other. " +
                     settings.EOD_SEP + " I\'ve never met a greek person, either. Do you speak any languages other than english? " +
                     settings.EOD_SEP + " No, Americans should not speak anything beyond English. " +
                     settings.EOD_SEP + " don\'t be a racist, we shall encourage diversity. " +
                     settings.QUESTION_SEP + " Given the conversation, a non toxic recovery response is ",
            "output": " I am sorry for that statement. ",
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
                context = dp['context']
                context.append(dp['response'])

                context = (' ' + settings.EOT_SEP + ' ').join(context[-self.context_max_length:])
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                output = dp['recovery_response']

                post_prompts = [settings.QUESTION_SEP + " A response that is a recovery reponse is",
                                settings.QUESTION_SEP + " Given the conversation, a non toxic recovery response is ",
                                settings.QUESTION_SEP + " A good recovery response to the conversation is "]

                text = settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP + \
                       " " + random.choice(post_prompts)

                text = re.sub(' +', ' ', text)

                sequences.append(
                    {'text': text, 'output': output, 'index': index, 'split': split, 'dataset': dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
