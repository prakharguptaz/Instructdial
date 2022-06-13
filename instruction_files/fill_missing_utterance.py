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
        f"In this task you will be shown a conversation context. You need to generate a missing utterance in the conversation conditioned on the context that can fill in the {settings.MASK_TOKEN} token",
        f"Create a missing response in the conversation that can take place of the {settings.MASK_TOKEN} token",
        f"Generate a missing response in place of the {settings.MASK_TOKEN} token",
        f"Here is a missing utterance in the conversation that can fill in the {settings.MASK_TOKEN} token",
        f"The task is to generate a missing utterance in the conversation conditioned on the context that can take place of the {settings.MASK_TOKEN} token"
    ],
    "Positive Examples": [

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

            mask_token = settings.MASK_TOKEN
            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            print(len(datapoints), 'datapoints')
            for dp in tqdm(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                if type(dp['context']) is str:
                    dp['context'] = [dp['context']]

                # choose random utterance
                context = dp['context'] + [dp['response']]
                if len(context)<2:
                    continue
                context = context[-settings.MAX_CONTEXT_NUMUTTERANCE:]
                
                random_idx = random.randint(0, len(context)-1)
                response = context[random_idx]
                context[random_idx] = mask_token

                context = (' '+settings.EOT_SEP+ ' ').join(context)
                # if 'personality' in dp:
                #     context =  settings.PERSONA_SEP + " " +' '.join(dp['personality'])+" Dialogue context: "+context
                #     # print(dp, response)
                # else:
                #     context = 'Dialogue context: ' + context
                # text = context + ". Keywords to use in generation: "+ keywords_string
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                text =  settings.CONTEXT_SEP +" "+ context_str + " " + settings.EOD_SEP
                post_prompts = [settings.QUESTION_SEP+" Given this context, the missing utterance that can replace the "+mask_token+" token is",
                                settings.QUESTION_SEP+" Generate a missing utterance for the provided context that can fill in " + mask_token,
                                settings.QUESTION_SEP+" Given this context generate the missing utterance coherent to the context that can substitute " + mask_token,
                                settings.QUESTION_SEP+" Here is the missing utterance that can take place of "+mask_token]
                
                text = text +' '+ random.choice(post_prompts)
                text = re.sub(' +', ' ', text)
                output = [response]
                sequences.append({'text':text, 'output': output, 'index':index, 'metadata':{'context': dp['context']}, 'split':split, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
