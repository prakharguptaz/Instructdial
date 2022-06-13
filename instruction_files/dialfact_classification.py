from instruction_files.generator_class import GeneratorBasic

import string
import json
import random
from string import Template, ascii_uppercase
import os
import re
from collections import Counter, defaultdict
import settings
from utils.common import get_options_string, get_alphabetwithoptions_string


instruction_dict = {
	"id": "intent_classification",
    "Source": [
        "intent",
    ],
    "Definitions": ["You will be given some dialogue context, a response and some evidence documents, and you need to find if the evidence supports or refutes the information in the response or does not have enough information for verificaiton",
                    "Given a conversation context, response and evidence documents, choose if the evidence supports, refutes (or disproves) or does not have enough information to verify the response",
                    "In this task you will be shown some evidence and a conversation and its response. Choose if the evidence supports, refutes or does not have enough information to verify the information in the response",
                    "You will be given some dialogue, a response, and some evidence documents. Find if the evidence supports or refutes the information in the response, or if there is not enough information to verify.",
                    "Given some dialogue, a response, and some evidence, you will need to determine if the evidence supports, refutes, or has insufficient information to verify the response.",
                    "Given some dialogue, a response, and some evidence, you will need to determine if the evidence supports, refutes or disproves, or has insufficient information to verify the response.",
                    ],
    "Positive Examples": [
            
    ]
}


# def list_tostring(classes):
#     assert type(classes) == list
#
#     return '||||'.join(classes)

class Generator(GeneratorBasic):
    def __init__(self, args, taskconfig, data_readers):
        self.idx = 0
        self.args = args
        self.taskconfig = taskconfig
        if 'max_data' in self.taskconfig:
            self.max_data = self.taskconfig['max_data']
        else:
            self.max_data = args.max_data
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
            datapoints = []
            split = dataset_reader.split
            dp = dataset_reader.get_next()
            while dp is not None:
                # print(dp)
                iterator_index+=1
                dp = dataset_reader.get_next()
                if dp and 'index' not in dp:
                    dp['index'] = iterator_index
                if dp:
                    datapoints.append(dp)
                    dp['split'] = split
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data))

            definitions = instruction_dict['Definitions']
            
            for dp in datapoints:
                # classes = dataset_reader.labels
                classes = ['supports', 'refutes', 'not enough information']
                random.shuffle(classes)
                dplabel = dp['label']
                if dplabel=='not enough info':
                    dplabel = 'not enough information'
            # mapping = {'classes':list_tostring(classes)}
                # instruction_sent = f'. The possible classes are: {settings.OPTION_TOKEN} $classes'
                # mapped_instruction = Template(instruction_sent).substitute(**mapping)
                mapped_instruction = get_alphabetwithoptions_string(classes)
                answer_idx = classes.index(dplabel)
                output = ascii_uppercase[answer_idx]
                candidate_options = []
                for option, candidate in zip(ascii_uppercase, classes):
                    candidate_options.append(f'{option}')
                # directly selecting does not work well for this task, option index works better
                # mapped_instruction = get_options_string(classes)
                # output = dplabel

                response_string, context_string = None, None
                context_list = []

                context_string = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                response_string = dp['response']
                # context_list = dp['context']
                evidence_list = dp['evidence_list']
                evidence_list = ['title: ' + ev[0] + ' text: ' + ev[2] for ev in evidence_list[:5]]
                doc = (' ' + settings.CLASS_SEPARATOR + ' ').join(evidence_list)
                doc = re.sub("  +", " ", doc)
                if len(doc.split()) > settings.MAX_DOCUMENT_LENGTH:
                    doc = ' '.join(doc.split())[:settings.MAX_DOCUMENT_LENGTH]
                context_string = ' '.join(context_string.split()[-settings.MAX_DIALOGUE_LENGTH:])
                # if dataset_reader.name == ''dnli
                text = f' {settings.EVIDENCE_SEP} {doc} '+  settings.CONTEXT_SEP+' ' +context_string +' '+settings.RESPONSE_SEP+' ' + response_string +' ' + settings.EOD_SEP + ' '+ mapped_instruction
                post_prompts = [
                                settings.QUESTION_SEP+'. The predicted class based on the context, response and evidence is',
                                settings.QUESTION_SEP+'. Choose the most possible class based on the response and the evidence',
                                settings.QUESTION_SEP+'. The best option given the context and the response is',
                                settings.QUESTION_SEP+". The best option among the provided classes based on the response and the evidence is"
                                ]

                text = text + ' ' + random.choice(post_prompts)
                text = re.sub(' +', ' ', text)
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')

                sequences.append({'input':text, 'outputs': [output], 'metadata':{'context': dp['context'], 'knowledge':doc}, 'index':index, 'split':split, 'classes_in_options':candidate_options, 'candidates': classes, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
