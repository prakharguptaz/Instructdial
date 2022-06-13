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

random.seed(123)

instruction_dict = {
    "id": "question_generation",
    "Definitions": [
        "In this task you will be shown a conversation. You need to generate a question asking for information that was mentioned in the conversation",
        "Using the given conversation, generate a question about the conversation",
        "Return a question about the given conversation "],
    "Positive Examples": [
        {
            "text": "[CONTEXT] ADAM FRANK, BYLINE: You know, when you can go get your, you know, your eggs any time of day, then you've sort of been lifted out of this very essential connection between you and the world, where it used to be that market day  - the market was only open at certain times, and often the market was only open certain days. NEAL CONAN ...",
            "output": "What were the two harvest-based calendars mentioned by the Jewish educator?",
        }
    ]
}


post_prompts = [settings.QUESTION_SEP + ' Given this context, generate an appropriate question',
                settings.QUESTION_SEP + ' What is a question for this context?',
                settings.QUESTION_SEP + ' A good question from the conversation is']


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

            print(len(datapoints), 'datapoints')

            idx = 0
            for dp in tqdm(datapoints):
                
                split = dp['split']
                context = dp['qg_context']
                if type(context) is str:
                    context_str = (' '+settings.EOT_SEP+' ').join(context.split("\n")) + " " + settings.EOD_SEP
                else:
                    context_str = (' '+settings.EOT_SEP+' ').join(context[-settings.MAX_CONTEXT_NUMUTTERANCE:]) + " " + settings.EOD_SEP

                post_prompts = ["What should we ask about this conversation?", "A question about this conversation is", "Here is a question realted to the dialogue"]
                post_prompt = random.choice(post_prompts)
                context_str = " ".join(context_str.split()[-settings.MAX_DIALOGUE_LENGTH:])
                # input_text = f" {context_str}. {post_prompt}"
                text = f"{settings.CONTEXT_SEP} {context_str} {settings.QUESTION_SEP} {post_prompt}"

                # text = text + ' ' + random.choice(post_prompts)
                
                output = dp['question']
                sequences.append({'input': text, 'output': output, 
                                'index': idx, 'split': split, 'dataset': dataset_reader.name})
                    
                idx += 1
            print(sequences[0])
        
        return sequences, instruction_dict

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
