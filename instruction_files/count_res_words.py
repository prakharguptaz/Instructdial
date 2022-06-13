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
	"id": "wow",
    "Definitions": [
        "In this task you will be shown a conversation context. You need to generate length of the final response to the conversation conditioned on the context.",
        "Provide length of a response to the conversation",
        "Count the number of words in the final response to the conversation",
        "Count the number of words in the final utterance in the conversation",
        "Generate length of a response to the dialogue",
        "Output the length of the response to the conversation",
        "Given the conversation, return length of the final response"
                    ],
    "Positive Examples": [
        {
          "text": "[CONTEXT] Do you know Manchester United F.C.? [ENDOFTURN] Is that a soccer team? [ENDOFTURN] Yes I believe it is . [ENDOFTURN] I really don't follow soccer. Are they good? [ENDOFDIALOGUE] [QUESTION] Given this context generate a response coherent to the context",
          "output": [
            #"I am not much of a soccer follower either, will have to check them out. "
            '15'
          ],
          "index": 36608,
          "metadata": {
            "context": [
              "Do you know Manchester United F.C.?",
              "Is that a soccer team?",
              "Yes I believe it is .",
              "I really don't follow soccer. Are they good?"
            ]
          },
          "split": "train",
          "dataset": "opendialkg"
        },
        {
          "text": "[CONTEXT] Yo! [ENDOFTURN] yo wassup! [ENDOFTURN] I am a ford mustang. i love them [ENDOFDIALOGUE] [QUESTION] Here is a response which is a good follow-up to the context",
          "output": [
            #"i would love to own a rolls royce ghost",
            '9'
          ],
          "index": 3663,
          "metadata": {
            "context": [
              "Yo!",
              "yo wassup!",
              "I am a ford mustang. i love them"
            ]
          },
          "split": "train",
          "dataset": "convai2"
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

def list_tostring(classes):
    assert type(classes) == list
    
    return '||||'.join(classes)

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
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data*5))

            definitions = instruction_dict['Definitions']

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            print(len(datapoints), 'datapoints')
            for dp in tqdm(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                if type(dp['context']) is str:
                    dp['context'] = [dp['context']]
                context = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                response = dp['response']
                # if 'personality' in dp:
                #     context =  settings.PERSONA_SEP + " " +' '.join(dp['personality'])+" Dialogue context: "+context
                #     # print(dp, response)
                # else:
                #     context = 'Dialogue context: ' + context
                # text = context + ". Keywords to use in generation: "+ keywords_string
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                text =  settings.CONTEXT_SEP +" "+ context_str + " " + settings.RESPONSE_SEP + " " + response + " " + settings.EOD_SEP
                post_prompts = [settings.QUESTION_SEP+" Given this context, the length of the final response is",
                                settings.QUESTION_SEP+" Generate length of the response for the provided context",
                                settings.QUESTION_SEP+" Generate length of the final utterance in the provided conversationi",
                                settings.QUESTION_SEP+" Given this context what is the length of the final response to the context",
                                settings.QUESTION_SEP+" Here is length of the response in the context"]
                
                text = text +' '+ random.choice(post_prompts)
                text = re.sub(' +', ' ', text)
                output = [len(response.split())]
                sequences.append({'text':text, 'output': output, 'index':index, 'metadata':{'context': dp['context'], 'response':dp['response']}, 'split':split, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
