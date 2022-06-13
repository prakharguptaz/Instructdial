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
    "Definitions": ["In this task you will be shown a conversation and a wikipedia text. You need to generate a response to the conversation based on the conversation and the provided snippet.",
                    "Read the dialogue and the wikipedia text to generate a response",
                    "Return a response to the provided dialogue using some wikipedia text"],
    "Positive Examples": [
        {
            "text": "'[DOCUMENT]': Game design __knowledge__ Game Design is the art of applying design and aesthetics to create a game for entertainment or for educational, exercise, or experimental purposes. [CONTEXT]: I want to learn how to design video games, trying to read and learn about it . [EOT]",
            "output": "It would be so interesting to help build a game for entertainment or educational purposes",
            "index": 180,
            "split": "train",
            "dataset": "wow"
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
            print(dataset_reader.name, len(datapoints))
            definitions = instruction_dict['Definitions']

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            for dp in datapoints:
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                context = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-self.context_max_length:])
                knowledge = dp['knowledge']
                if type(knowledge) is list: # in wow, only forst is correct knolwedge
                    knowledge = dp['knowledge'][0]
                if len(knowledge.split())>settings.MAX_KNOWLEDGE_LENGTH:
                    knowledge = ' '.join(knowledge.split())[:settings.MAX_KNOWLEDGE_LENGTH]
                response = dp['response']
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                if 'no_fact' in knowledge or len(knowledge)<8:
                    continue
                # text =  "[knowledge text]: " + knowledge + ' [dialogue context]: ' + context
                text =  settings.WIKI_SEP + " " + knowledge +" "+ settings.CONTEXT_SEP +" "+ context_str + " " + settings.EOD_SEP
                post_prompts = [settings.QUESTION_SEP+" Given this context and knowledge, the response is",
                                settings.QUESTION_SEP+" Generate a response with the provided context which contains the provided knowledge",
                                settings.QUESTION_SEP+" Given this context generate a response which has the given knowledge",
                                settings.QUESTION_SEP+" Here is a response which contains the given knowledge"]

                text = text +' '+ random.choice(post_prompts)
                text = re.sub(' +', ' ', text)
                output = response
                sequences.append({'text':text, 'output': output, 'metadata':{'context':dp['context'], 'knowledge':knowledge}, 'index':index, 'split':split, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
