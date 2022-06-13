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
    "id": "graph-based-generation",
    "Definitions": [
        "In this task you will be shown a conversation context and sets of triplets consisting of subject, object, and relation. You need to generate a response to the conversation based on the context and the triplets.",
        "Given a set of triplets -- subject, object, relation -- provide a response to a conversation using the triplets ",
        "Return a response to a conversation using information from a set of triplets"],
    "Positive Examples": [
        {
            "text": settings.GRAPH_SEP + " The subject is Iron Man, relation: starred_actors, object: Robert Downey Jr " +
                    settings.CONTEXT_SEP + " Do you like Iron Man " +
                    settings.EOD_SEP + " " +
                    settings.QUESTION_SEP + " Given this context and triplets, the response is ",
            "output": "Sure do! Robert Downey Jr. is a favorite.",
            "index": 1,
            "split": "train",
            "dataset": "opendialkg"
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
            self.context_max_length = 5
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

                graph = dp['graph']
                if len(graph.split()) > settings.MAX_DOCUMENT_LENGTH:
                    graph = ' '.join(graph.split())[:settings.MAX_DOCUMENT_LENGTH]

                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-self.context_max_length:])
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                post_prompts = [settings.QUESTION_SEP + " The response is ",
                                settings.QUESTION_SEP + " Given this context and triplets, the response is ",
                                settings.QUESTION_SEP + " A good response using the triplets is "]

                output = dp['response']

                text = settings.GRAPH_SEP + " " + graph + " " + \
                       settings.CONTEXT_SEP + " " + context_str + " " + \
                       settings.EOD_SEP + " " + random.choice(post_prompts)

                text = re.sub(' +', ' ', text)
                sequences.append( {'text': text,
                                   'output': output,
                                   'metadata':{'context':dp['context'], 'graph':graph},
                                   'index': index,
                                   'split': split,
                                   'dataset': dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
