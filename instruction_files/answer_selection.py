from instruction_files.generator_class import GeneratorBasic
from utils import extraction

import string
import json
import random
from string import Template, ascii_uppercase
import os
from collections import Counter, defaultdict
import settings
from tqdm import tqdm
import re
from utils.common import get_options_string, get_alphabetwithoptions_string

random.seed(123)

instruction_dict = {
    "id": "answer_selection",
    "Definitions": [
        "In this task you will be shown a conversation with an optional document. You need to consider the conversation context and select the option corresponding to the answer to the question.",
        "You will be given a conversation and optionally a document. Choose an answer to the question given the dialogue context.",
        "You will be given a document and a conversation. You will need to read the conversation and select the option that answers the question.",
        "You will be given a conversation with an optional document. You will need to read the conversation and select the best answer to the question.",
        "Given a conversation and a document, you need to read the conversation and document to select the answer that best answers the question.",
        "Read the conversation and the optional document and select the answer that best responds to the question.",
        "Consider the context of the conversation and a document and choose an answer accordingly."],
    "Definitions-timedial":[
        "In this task you will be provided a conversation. Select the option which can take the value of <MASK> in the dialog context"
        "In this task you will be shown a conversation. You need to consider the conversation context and select the option which can substitute the <MASK> in the conversation.",
        "You will be given a conversation. Choose an option that can take the the value of <MASK> given the dialogue context."
    ],
    "Positive Examples": [
        {
            "input": "[DOCUMENT] Jessica went to sit in her rocking chair. Today was her birthday and she was turning 80 [CONTEXT]Who had a Birthday? Jessica How old would she be? [EOT] Which is the best answer? [SEP] 80 [EOT] park [EOT] implies",
            "output": "80",
        }
    ]
}


# def list_tostring(classes):
#     assert type(classes) == list
#     lenc = len(classes)
#     if len(classes)<2:
#         return ' '.join(classes)
#     elif len(classes)==2:
#         return classes[0] + ' and ' + classes[1]
#     else:
#         return ', '.join(classes[:-1]) + ' and ' + classes[-1]

# def list_tostring(candidates):
#     assert type(candidates) == list
    
#     candidate_with_option = []
#     for option, candidate in zip(ascii_uppercase, candidates):
#         candidate_with_option.append(f'{option}: {candidate}')
#     return ' |||| '.join(candidate_with_option)

mask_selection_datasets = [
    'timedial'
]

post_prompts = [settings.QUESTION_SEP + " Select the best option from the candidate answer options",
                settings.QUESTION_SEP + " Choose the best answer index", 
                settings.QUESTION_SEP + " What is the best candidate answer?", 
                settings.QUESTION_SEP + " The best option among the answers is"]

mask_prompts = [settings.QUESTION_SEP + " Select an option which will fill in the <MASK>",
                settings.QUESTION_SEP + " What is the best candidate answer for the <MASK>",
                settings.QUESTION_SEP + " The best answer filling in the <MASK> is"]


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
            
            print(datapoints[-1])
            print(len(datapoints), 'datapoints')
                
            answer_set = [d['answer'] for d in datapoints]

            idx = 0
            for dp in tqdm(datapoints):
                context, answer = dp['context'], dp['answer']
                if type(context) is not list:
                    context = [context]
                context_str = f' {settings.EOT_SEP} '.join(context[-settings.MAX_CONTEXT_NUMUTTERANCE:])
                context_str = ' '.join(context_str.split()[-settings.MAX_DIALOGUE_LENGTH:])

                negative = []
                while len(negative) < 3:
                    r = random.choice(answer_set)
                    if r == answer or r in negative:
                        continue
                    else:
                        negative.append(r)
                
                answer_idx = random.randint(0, 3)
                candidates = negative[:answer_idx] + [answer] + negative[answer_idx:]
                assert candidates[answer_idx] == answer

                if 'mutual' in dataset_reader.name or 'timedial' in dataset_reader.name:
                    candidates = dp['options']
                    answer_idx = candidates.index(answer)

                
                answer_str = get_alphabetwithoptions_string(candidates)
                if 'document' in dp:
                    document = dp['document']
                    document = ' '.join(document.split())[:settings.MAX_KNOWLEDGE_LENGTH] 
                    text = f"{settings.WIKI_SEP} {document} {settings.CONTEXT_SEP} {context_str} {settings.EOD_SEP} These are the candidate answers: {answer_str}"

                else:
                    text = f"{settings.CONTEXT_SEP} {context_str} {settings.EOD_SEP} These are the candidate answers: {answer_str}"

                if dataset_reader.name in mask_selection_datasets:
                    text = text + ' ' + random.choice(mask_prompts)
                else:
                    text = text + ' ' + random.choice(post_prompts)

                output = ascii_uppercase[answer_idx]
                text = re.sub(' +', ' ', text)

                candidate_options = []
                for option, candidate in zip(ascii_uppercase, candidates):
                    candidate_options.append(f'{option}')
                sequences.append({'input': text, 'output': output, 
                                     'metadata': {'context':dp['context']},
                                    'index': idx, 'split': dp['split'], 'classes_in_options':candidate_options, 'candidates': candidates, 'dataset': dataset_reader.name})
                idx += 1
            print(sequences[-1])

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
