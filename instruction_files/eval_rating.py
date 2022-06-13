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
    "id": "evaluation_binary",
    "Definitions": [
        "In this task you will be shown a conversation followed by a response and score range. You need to give a appropriate rating to the response for the stated criteria.",
        "Given a conversation, a response to the conversation, and a score range, return a rating to the response within in score range for the stated criteria",
        "Return a score in the provided range for the given criteria which scores the response to a conversation "],
    "Positive Examples": [
        {
            "input": "[CONTEXT] this is @sprint great service nothing like 3g speeds in 2017 that doesn 't even work . lol <URL> [EOT] Please give a rating ranging from 1 to 5 about the overall to the following response: please dm us your account information and we will be happy to assist you ",
            "output": "3.5",
        }
    ]
}


def list_tostring(classes):
    assert type(classes) == list
    lenc = len(classes)
    if len(classes)<2:
        return ' '.join(classes)
    elif len(classes)==2:
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
            split = dataset_reader.split
            
            datapoints = []
            dp = dataset_reader.get_next()
            while dp is not None:
                dp = dataset_reader.get_next()

                if dp:
                    datapoints.append(dp)
                    dp['split'] = split
            
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data))
            definitions = instruction_dict['Definitions']
            
            print(datapoints[0])
            print(len(datapoints), 'datapoints')
            response_set = [dp['response'] for dp in datapoints]
            idx = 0
            print(split)
            for dp in tqdm(datapoints):
                eval_dataset_name  = dp['dataset_name']
                if (split=='train' or split=='all') and eval_dataset_name not in ['humod']:
                    continue

                text = ''
                
                context, response = ' '.join(dp['context']), dp['response']
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                
                min_score, max_score = dp['score_min'], dp['score_max']

                for quality in dp['qualities']:
                    score = dp['score'][quality]

                    text = f"{settings.CONTEXT_SEP} {context_str} {settings.EOD_SEP} {settings.QUESTION_SEP} Please give a rating ranging from {min_score} to {max_score} about the {quality} criteria to the following response: {response}"
                    if dp['persona'] is not None:
                        text = settings.PERSONA_SEP + " " + ' '.join(dp['persona']) +" "+ text
                    if dp['knowledge'] is not None:
                        text = settings.WIKI_SEP + " " + (dp['knowledge']) +" "+ text                    
                        score = f'{float(score):.2f}'.rstrip('0').rstrip('.')
                    output = score
                    text = re.sub(' +', ' ', text)

                    sequences.append({'input': text, 'outputs': output, 'metadata': {'human_rating':dp['score'], 'eval_dataset_name':eval_dataset_name},
                                        'index': idx, 'split': dp['split'], 'dataset': dataset_reader.name})
                idx += 1
            print(sequences[-1])

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
