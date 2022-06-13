from instruction_files.generator_class import GeneratorBasic

import string
import json
import random
from string import Template
import os
import re
from collections import Counter, defaultdict
import settings
from utils.common import get_options_string, get_alphabetwithoptions_string

instruction_dict = {
	"id": "intent_present",
    "Source": [
        "intent",
    ],
    "Definitions": ["You will be shown a dialogue utterance and you need to answer yes or no if the intent is present in the utterance. ",
                    "You will be given some conversation response and you need to decide if the intent is in the utterance.",
                    "Decide if an intent is in the provided dialog utterance "],
    "Positive Examples": [
        {
            "input": "[RESPONSE] list the three earliest flights from atlanta to philadelphia on wednesday [ENDOFDIALOGUE]. The possible options are: [OPTIONS] yes||||no [QUESTION]. Is the intent flight correct for the response?. Choose among [OPTIONS] yes||||no",
            "outputs": [
                "yes"
            ],
            "index": 4730,
            "split": "train",
            "dataset": "atis"
        },
        {
            "input": "[RESPONSE] please list all flights that leave denver before noon on northwest airlines [ENDOFDIALOGUE]. The possible options are: [OPTIONS] yes||||no [QUESTION]. Does the provided intent match the response? The intent is flight. Choose among the options [OPTIONS] yes||||no",
            "outputs": [
                "no"
            ],
            "candidates": [
                "yes",
                "no"
            ],
            "classes_in_options": [
                "yes",
                "no"
            ],
            "index": 4523,
            "split": "train",
            "dataset": "atis"
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
        self.output_folder = args.tasks_output_folder
        self.out_file = self.taskconfig['task_name']
        self.data_readers = data_readers
        self.examples = []

    def generate_data(self):
        print('number of datareaders:', len(self.data_readers))
        sequences = []
        for d, dataset_reader in enumerate(self.data_readers):
            print(dataset_reader.name)
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

            yesnooptionstring = get_options_string(['yes', 'no'])
            candidates = ['yes', 'no']
            mapped_instruction = '. The possible options are: ' + yesnooptionstring

            for dp in datapoints:


                # num_classes = random.randint(2, len(dataset_reader.intent_classes))
                # num_classes = min(num_classes, 10)
                # classes = random.sample(dataset_reader.intent_classes, num_classes) + [dp['intent_label']]
                # random.shuffle(classes)
                # mapping = {'classes':list_tostring(classes)}
                # instruction_sent = '. The possible intents are: $classes'
                # mapped_instruction = Template(instruction_sent).substitute(**mapping)
                context = dp.get('context', '')
                if context !='' and type(context) is list:
                    context = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])


                if context!='':
                    text = settings.CONTEXT_SEP + ' ' + context + ' '+settings.RESPONSE_SEP + ' ' + dp['response']+ ' ' + settings.EOD_SEP + mapped_instruction
                else:
                    text = settings.RESPONSE_SEP + ' ' + dp['response']+ ' ' + settings.EOD_SEP + mapped_instruction
                output = 'yes'
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                post_prompts = [settings.QUESTION_SEP+'. Is the intent ' + dp['intent_label'] + ' present in the response?', settings.QUESTION_SEP+'. The intent ' + dp['intent_label'] + ' is present in the response?', settings.QUESTION_SEP+'. Does the provided intent match the response? The intent is ' + dp['intent_label'], settings.QUESTION_SEP+'. Does the provided intent match the response? The intent is ' + dp['intent_label'] , settings.QUESTION_SEP+'. Is the intent ' + dp['intent_label'] + ' correct for the response?']

                text = text + ' ' + random.choice(post_prompts)
                sequences.append({'input':text, 'outputs': [output], 'candidates':candidates, 'classes_in_options':candidates, 'metadata':{'context':dp.get('context', ''), 'response':dp.get('response',''), 'intent':dp['intent_label']}, 'index':index, 'split':split, 'dataset':dataset_reader.name})

                random_intent = random.choice(dataset_reader.intent_classes)
                while random_intent==dp['intent_label']:
                    random_intent = random.choice(dataset_reader.intent_classes)
                # text = settings.RESPONSE_SEP + ' ' + dp['response'] + ' ' + settings.EOD_SEP

                if context!='':
                    text = settings.CONTEXT_SEP + ' ' + context + ' '+settings.RESPONSE_SEP + ' ' + dp['response']+ ' ' + settings.EOD_SEP + mapped_instruction
                else:
                    text = settings.RESPONSE_SEP + ' ' + dp['response']+ ' ' + settings.EOD_SEP + mapped_instruction
                output = 'no'
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                post_prompts = [settings.QUESTION_SEP+'. Is the intent: ' + random_intent + ' present in the response?',
                                settings.QUESTION_SEP+'. The intent ' + random_intent + ' is present in the response?',
                                settings.QUESTION_SEP+'. Does the provided intent match the response? The intent is ' + random_intent,
                                settings.QUESTION_SEP+'. Does the provided intent match the response? The intent is: ' + random_intent,
                                settings.QUESTION_SEP+'. Is the intent ' + random_intent + ' correct for the response?']

                text = text + ' ' + random.choice(post_prompts)
                text = re.sub(' +', ' ', text)

                sequences.append({'input':text, 'outputs': [output], 'candidates':candidates, 'classes_in_options':candidates, 'metadata':{'context':dp.get('context', ''), 'response':dp.get('response',''), 'intent':random_intent}, 'index':index, 'split':split, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
