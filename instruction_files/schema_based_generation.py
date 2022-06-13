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
    "Definitions": ["Read the dialogue and the schema to generate a response",
                    "Return a response to the conversation based on information from the provided schema",
                    "Generate a response to the dialog using the provided schema information"],
    "Positive Examples": [
        {"text":
             settings.SCHEMA_SEP + " {'label': 'Open circuit on run?', 'terminal': False, 'utterance': \"Is there an open circuit with your car'\'s key in 'run/ON' position?\"} " +
             settings.CONTEXT_SEP + " My car is experiencing an electrical failure. I need it fixed pronto because I am taking a road trip soon. " +
             settings.EOD_SEP + " " +
             settings.QUESTION_SEP + " Given this context and schema, the response is ",
         "output": "'When your car key is in the \'run/ON\' position, is there an open circuit?'",
         "index": 11,
         "split": "train",
         "dataset": "flodial"
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

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            for i, dp in enumerate(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')

                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-self.context_max_length:])
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                post_prompts = [settings.QUESTION_SEP + " The response is ",
                                settings.QUESTION_SEP + " Given this context and schema the response is "]

                text = settings.SCHEMA_SEP + " " + dp['schema'] + " " \
                       + settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP \
                       + " " + random.choice(post_prompts)

                output = dp['response']
                text = re.sub(' +', ' ', text)
                sequences.append({'text': text,
                                  'output': output,
                                  'index': index,
                                  'split': split,
                                  'dataset': dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
