from instruction_files.generator_class import GeneratorBasic

import string
import json
import random
from string import Template
import os
import settings
from collections import Counter, defaultdict

random.seed(123)
import re

instruction_dict = {
    "id": "db-based-generation",
    "Definitions": ["Read the dialogue and to generate the belief state",
                    "Generate the belief state given the dialogue",
                    "Predict the current belief state from the dialogue"],
    "Positive Examples": [
        {
            "text": settings.CONTEXT_SEP + " there are 21 guesthouses which offer free parking . which area do you prefer to stay in ? " +
                    settings.EOT_SEP + " i am open to any area , but the hotel should definitely have only 1 star . " +
                    settings.EOD_SEP + " " +
                    settings.QUESTION_SEP + " Given this context, the belief is ",
            "output": "hotel-parking : yes , hotel-type : guest house",
            "index": 5714,
            "split": "train",
            "dataset": "multiwoz"
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
        self.context_max_length = settings.MAX_CONTEXT_NUMUTTERANCE

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
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data))

            definitions = instruction_dict['Definitions']

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            max_text_size = -1
            max_context_len = -1
            for i, dp in enumerate(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')

                post_prompts = [settings.QUESTION_SEP + " What is the belief state? ",
                                settings.QUESTION_SEP + " The belief state is ",
                                settings.QUESTION_SEP + " What is a the belief state for this dialogue? "]

                max_context_len = max(max_context_len, len(dp['context']))
                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-self.context_max_length:])
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                # context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][:])
                # context_str = ' '.join(context.split()[:])

                text = settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP + " " + random.choice(post_prompts)
                output = dp['state']
                text = re.sub(' +', ' ', text)
                max_text_size = max(len(text.split()),max_text_size)
                sequences.append(
                    {'text': text,
                     'output': output,
                     'index': index,
                     'split': split,
                     'dataset': dataset_reader.name})
        print('max_text_size', max_text_size)
        print('max_context_len', max_context_len)


        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
