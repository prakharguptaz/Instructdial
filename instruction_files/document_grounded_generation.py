from instruction_files.generator_class import GeneratorBasic

import string
import json
import random
from string import Template
import os
import settings
from collections import Counter, defaultdict
import re

random.seed(123)

instruction_dict = {
    "id": "document_grounded_generation",
    "Definitions": [
        "In this task you will be shown a conversation and facts from a document. You need to generate a response to the conversation based on the provided document facts.",
        "You will be shown a conversation and a document. Generate a response to the conversation using the provided document facts.",
        "Read the dialogue and the document text to generate a response",
        "Given a conversation and facts from a document, provide a response to the conversation which uses the document facts"],
    "Positive Examples": [
        {
            "text": settings.WIKI_SEP + " you must report a change of address to DMV within ten days of moving. " +
                    settings.CLASS_SEPARATOR + " That is the case for the address associated with your license, as well as all the addresses associated with each registered vehicle, which may differ. " +
                    settings.CONTEXT_SEP + " Hello, I forgot o update my address, can you help me with that? " +
                    settings.EOD_SEP + " " +
                    settings.QUESTION_SEP + " Given this context and document, the response is ",
            "output": "hi, you have to report any change of address to DMV within 10 days after moving. You should do this both for the address associated with your license and all the addresses associated with all your vehicles.",
            "index": 1,
            "split": "train",
            "dataset": "doc2dial"
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
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data))

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            max_text_size = -1
            for dp in datapoints:
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')

                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-self.context_max_length:])
                if type(dp['doc']) is list:
                    doc = (' ' + settings.CLASS_SEPARATOR + ' ').join(dp['doc'])
                else:
                    doc = dp['doc']
                doc = re.sub("  +", " ", doc)

                if len(doc.split()) > settings.MAX_DOCUMENT_LENGTH:
                    doc = ' '.join(doc.split())[:settings.MAX_DOCUMENT_LENGTH]
                output = dp['response']
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                text = settings.WIKI_SEP + " " + doc + " " + settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP
                post_prompts = [settings.QUESTION_SEP+" Given this context and provided document, the response is",
                                settings.QUESTION_SEP+" Generate a response to the provided context which contains the provided document content",
                                settings.QUESTION_SEP+" Given this context generate a response which has the given document knowledge",
                                settings.QUESTION_SEP+" Here is a response that contains the given document knowledge"]

                text = text +' '+ random.choice(post_prompts)
                text = re.sub(' +', ' ', text)
                max_text_size = max(len(text.split()),max_text_size)
                dat = {'text': text,
                                  'output': output,
                                  'index': index, 
                                  'metadata': {'document':doc, 'context':dp['context']},
                                  'split': split,
                                  'dataset': dataset_reader.name}
                sequences.append(dat)
        print('max_text_size', max_text_size)

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
