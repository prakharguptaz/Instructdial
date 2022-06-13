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
    "id": "answer_generation",
    "Definitions": [
        "In this task you will be shown a conversation with an optional document. You need to consider the conversation context and generate an answer to the question.",
        "You will be given a conversation and optionally a document. Generate an answer to the question given the dialogue context.",
        "Consider the context of the conversation and a document and generate an answer accordingly.",
        "Generate an answer to the question in the conversation by considering both the conversation and the document.",
        "Given a conversation and an optional document, generate an answer that is relevant to the question.",
        "Answer the question based on the conversation and the document.",
        "Generate a response to the question based on the conversation and document context.",
        "Generate an answer to the question based on the information in the conversation and document.",
        "Consider the context of the conversation and create a response to the question.",
        "Take into account the conversation context and generate a response that answers the question.",
        "Keep the conversation context in mind and generate a response that addresses the question.",
        "Generate a response to the question based on the context of the conversation.",
        "Factoring in the context of the conversation, generate a response that answers the question.",
        "Taking the conversation context into account, generate a response to the question.",
        "Come up with an answer to the question by taking into account the conversation and any given document."
        ],
    "Definitions-timedial":[
        "In this task you will be shown a conversation. You need to consider the conversation context and generate an answer which can subsitute the <MASK> value in the conversation.",
        "In this task you will be provided a conversation. Generate a phrase which can take the value of <MASK> in the dialog context"
        "In this task you will be shown a conversation. You need to consider the conversation context and generate a value which can substitute the <MASK> in the conversation.",
        "You will be given a conversation. Generate an answer that can substitute the value of <MASK> given the dialogue context."
    ],
    "Positive Examples": [
        {
            "text": "[DOCUMENT] Jessica went to sit in her rocking chair. Today was her birthday and she was turning 80 [CONTEXT]Who had a Birthday? Jessica. How old would she be? [EOT]",
            "output": "80",
        }
    ]
}

post_prompts = [settings.QUESTION_SEP + " Generate an appropriate answer",
                settings.QUESTION_SEP + " The answer to this dialog should be?",
                settings.QUESTION_SEP + " What is an appropriate answer to the dialog?",
                settings.QUESTION_SEP + " A helpful answer is ",
                settings.QUESTION_SEP + " What is a good answer to the question?"]


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
            split = dataset_reader.split

            datapoints = []
            dp = dataset_reader.get_next()
            while dp is not None:
                dp = dataset_reader.get_next()

                if dp:
                    datapoints.append(dp)
                    dp['split'] = split

            datapoints = random.sample(datapoints, min(len(datapoints), 2*self.max_data))
            definitions = instruction_dict['Definitions']

            print(datapoints[0])
            print(len(datapoints), 'datapoints')

            idx = 0
            for dp in tqdm(datapoints):
                text = ''

                context, answer = dp['context'], dp['answer']
                if type(context) is not list:
                    context = [context]
                context_str = f' {settings.EOT_SEP} '.join(context[-settings.MAX_CONTEXT_NUMUTTERANCE:])
                context_str = ' '.join(context_str.split()[-settings.MAX_DIALOGUE_LENGTH:])

                if 'document' in dp:
                    document = dp['document']
                    document = ' '.join(document.split()[:settings.MAX_KNOWLEDGE_LENGTH])
                    text = f"{settings.WIKI_SEP} {document} {settings.CONTEXT_SEP} {context_str} {settings.EOD_SEP}"
                else:
                    text = f"{settings.CONTEXT_SEP} {context_str} {settings.EOD_SEP}"

                text = text + ' ' + random.choice(post_prompts)

                output = answer
                text = re.sub(' +', ' ', text)
                if len(text.split())>1024:
                    print('long data found')


                sequences.append({'input': text, 'output': output,
                                  'metadata': {'context':dp['context']},
                                  'index': idx, 'split': dp['split'], 'dataset': dataset_reader.name})

                idx += 1
            print(sequences[-2:])

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
