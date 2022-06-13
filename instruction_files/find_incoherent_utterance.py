from instruction_files.generator_class import GeneratorBasic
from utils import extraction
from utils.common import get_options_string, get_alphabetwithoptions_string

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
    "Definitions": ["In this task you will be shown a conversation context. One utterance is incoherent with the rest of the conversation. You need to find the index of the incoherent utterance. Predict all correct if the conversation is correct",
                    "Given a conversation, find the index of an utterance that is incoherent with the rest of the conversation, otherwise predict all correct",
                    "Given a dialogue, find the index of an utterance that is incoherent with the rest of the dialogue, otherwise predict all correct",
                    "Predict the index of any incoherent utterance or response in this conversation. If no incoherent utterance present, predict all correct",
                    "Generate the index of any incoherent utterance or response in this dialogue. If no incoherent utterance is found, then predict all correct",
                    "Generate the index of an incoherent utterance in the provided conversation. Predict all correct if no incoherent utterance found",
                    "Given the conversation, return a index of an incoherent utterance, predict all correct if the conversation has no incoherent utterance"
                    ],
    "Positive Examples": [
        {
          "text": "[CONTEXT] Do you know Manchester United F.C.? [ENDOFTURN] Yes I believe it is . [ENDOFTURN] Is that a soccer team? [ENDOFTURN] I really don't follow soccer. Are they good? [ENDOFDIALOGUE] [QUESTION] Given this context generate a response coherent to the context",
          "output": [
            "1 and 2 "
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

            responses_bank = [dp['response'] for dp in datapoints]
            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            print(len(datapoints), 'datapoints')
            for dp in tqdm(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                if type(dp['context']) is str:
                    dp['context'] = [dp['context']]

                # choose random utterance
                context = dp['context'][:] + [dp['response']]
                if len(context)<3:
                    continue
                context = context[-settings.MAX_CONTEXT_NUMUTTERANCE:]

                if random.random()<0.2:
                    response_idx = 'all correct'
                    swapped_response = 'none'
                else:
                    random_idx = random.randint(1, len(context)-1)
                    response_idx = f'{random_idx}'
                    swapped_response = context[random_idx]
                    context[random_idx] = random.choice(responses_bank)

                context = [f"Index {str(i)} : " + x for i, x in enumerate(context)]
                candidates = [str(x) for x in list(range(0, len(context)))] + ['all correct']
                options_text = get_options_string(candidates)


                context = (' '+settings.EOT_SEP+ ' ').join(context)
                # if 'personality' in dp:
                #     context =  settings.PERSONA_SEP + " " +' '.join(dp['personality'])+" Dialogue context: "+context
                #     # print(dp, response)
                # else:
                #     context = 'Dialogue context: ' + context
                # text = context + ". Keywords to use in generation: "+ keywords_string
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                text =  settings.CONTEXT_SEP +" "+ context_str + " " + settings.EOD_SEP +" The possible indices are: " + options_text
                post_prompts = [settings.QUESTION_SEP+" Given this context, the index of the incoherent response is",
                                settings.QUESTION_SEP+" Generate the index of any inconsistent utternaces in the provided conversation",
                                settings.QUESTION_SEP+" Given this conversation generate the index of any incoherent utterance",
                                settings.QUESTION_SEP+" Here is the index of inconsistent utternace in the response"]
                
                text = text +' '+ random.choice(post_prompts)
                text = re.sub(' +', ' ', text)
                output = [str(response_idx)]
                sequences.append({'text':text, 'output': output, 'index':index, 'metadata':{'context': dp['context'], 'swapped_response':swapped_response}, 'split':split, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
