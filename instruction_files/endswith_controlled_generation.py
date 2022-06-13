from instruction_files.generator_class import GeneratorBasic
from utils import extraction

import string
import json
import random
from string import Template
import os
import re
from collections import Counter, defaultdict
import settings
from tqdm import tqdm

random.seed(123)

instruction_dict = {
	"id": "wow",
    "Definitions": ["In this task you will be shown a conversation context and a phrase. You need to generate a response to the conversation based on the context which ends with the provided phrase.",
                    "Read the dialogue and the provided phrase to generate a response which ends with the phrase provided.",
                    "Create a response given the dialogue and a phrase which ends with the provided phrase. "],
    "Positive Examples": [
        {
          "text": "[FINAL PHRASE] checks ? [CONTEXT] Are you through with your meal ? [ENDOFTURN] Yes , we are . Could we have the check , please ? [ENDOFTURN] Here is your check , 86 dollars in all . Can I take care of it here when you are ready ? [ENDOFDIALOGUE] [QUESTION] Given this context and final phrase, the response is",
          "output": "Do you accept checks ?",
          "metadata": {
            "context": [
              "Are you through with your meal ?",
              "Yes , we are . Could we have the check , please ?",
              "Here is your check , 86 dollars in all . Can I take care of it here when you are ready ?"
            ],
            "endswith": "checks ?"
          },
          "index": 58763,
          "split": "train",
          "dataset": "dailydialog"
        },
        {
          "text": "[FINAL PHRASE] . [CONTEXT] are you ready to go the concert ? [ENDOFTURN] yes . Should we go there by bus so we aren't late ? [ENDOFTURN] actually , why don't we go there by bike ? We could get stuck in traffic if we travel by bus in such hour . [ENDOFTURN] that's true . Cycling is good for our environment , too . Let me just get my helmet then . [ENDOFTURN] is your helmet comfortable ? [ENDOFTURN] not really , but I liked the design , so I got it . [ENDOFTURN] maybe you should think about getting a round helmet ; they're better . [ENDOFDIALOGUE] [QUESTION] Given this context and final phrase, the response is",
          "output": "I'll think about it .",
          "metadata": {
            "context": [
              "are you ready to go the concert ?",
              "yes . Should we go there by bus so we aren't late ?",
              "actually , why don't we go there by bike ? We could get stuck in traffic if we travel by bus in such hour .",
              "that's true . Cycling is good for our environment , too . Let me just get my helmet then .",
              "is your helmet comfortable ?",
              "not really , but I liked the design , so I got it .",
              "maybe you should think about getting a round helmet ; they're better ."
            ],
            "endswith": "."
          },
          "index": 13375,
          "split": "train",
          "dataset": "dailydialog"
        }
    ]
}


def get_endphrase(response):
    response_words = response.split()
    max_startwords = min(5,int(len(response_words)/2))
    if max_startwords<1:
        return ''
    num_wordsinphrase = random.randint(1,max_startwords)

    return ' '.join(response_words[-num_wordsinphrase:])


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
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data*5))

            definitions = instruction_dict['Definitions']

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            print(len(datapoints), 'datapoints')
            for dp in tqdm(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                context = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                response = dp['response']
                phrase = get_endphrase(response)
                # if 'personality' in dp:
                #     context =  settings.PERSONA_SEP + " " +' '.join(dp['personality'])+" Dialogue context: "+context
                # else:
                #     context = '[dialogue context]: ' + context

                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                text =  settings.FINALPHRASE_SEP + " " + phrase +" "+ settings.CONTEXT_SEP +" "+ context_str + " " + settings.EOD_SEP 
                post_prompts = [settings.QUESTION_SEP+" Given this context and final phrase, the response is",
                                settings.QUESTION_SEP+" Generate a response with the provided context which ends with the provided final phrase",
                                settings.QUESTION_SEP+" Given this context generate a response which ends with the given final sentence",
                                settings.QUESTION_SEP+" Here is a response which ends with the given final phrase"]
                
                text = text +' '+ random.choice(post_prompts)
                output = response
                sequences.append({'text':text, 'output': output, 'metadata':{'context': dp['context'], 'endswith':phrase}, 'index':index, 'split':split, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
