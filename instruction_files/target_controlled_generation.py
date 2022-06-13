from instruction_files.generator_class import GeneratorBasic

import string
import json
import random
from string import Template
import os
import re
import settings
from collections import Counter, defaultdict
random.seed(123)

instruction_dict = {
	"id": "wow",
    "Definitions": ["In this task you will be shown a dialogue context and a target sentence. Generate a response to the dialogue context so that the generated response is a smooth transition to the target sentence.",
                    "Read the dialogue and the target sentence to generate a response which is a good bridge to the target sentence provided.",
                    "Generate a text which connects the dialogue context with the target sentence."],
    "Positive Examples": [
        {
            "text": "[target sentence]: i love chocolate. [dialogue context]: i love walking in the park.",
            "output": "Park is a good place for picnic. And for the picnic dessert I love chocolate.",
            "index": 1553,
            "split": "train",
            "dataset": "otters"
        },
        {
            "text": "[target sentence]: i love reading mysteries in my free time. [dialogue context]: i love playing tennis.",
            "output": "i like to play tennis and read mysteries.",
            "index": 1669,
            "split": "train",
            "dataset": "otters"
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
            dataset_reader.idx=0
            iterator_index = 0
            split = dataset_reader.split
            datapoints = []
            dp = dataset_reader.get_next()
            while dp is not None:
                # print(dp)
                iterator_index+=1
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
            for dp in datapoints:
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                # context = ' [EOT] '.join(dp['context'][-self.context_max_length:])
                context = dp['context']
                knowledge = dp['target']
                if type(knowledge) is list:
                    knowledge = ' '.join(knowledge)
                if len(knowledge.split())>settings.MAX_KNOWLEDGE_LENGTH:
                    knowledge = ' '.join(knowledge.split())[:settings.MAX_KNOWLEDGE_LENGTH]
                response = dp['response']
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                text =  settings.TARGET_SEP + " " + knowledge +" "+ settings.CONTEXT_SEP +" "+ context_str + " " + settings.EOD_SEP 
                post_prompts = [settings.QUESTION_SEP+" Given this context and target sentence, a response which is a good bridge to the target sentence provided is",
                                settings.QUESTION_SEP+" Generate a response which when prepended to target sentence is a good transition to the provided target sentence.",
                                settings.QUESTION_SEP+" Given this context and a target sentence, generate a response which serves as a good transition to the target sentence",
                                settings.QUESTION_SEP+" Here is a response which is a good transition to the target sentence"]
                
                text = text +' '+ random.choice(post_prompts)
                text = re.sub(' +', ' ', text)
                output = response
                sequences.append({'text':text, 'output': output, 'index':index, 'metadata':{'context': dp['context'], 'target':dp['target']}, 'split':split, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
