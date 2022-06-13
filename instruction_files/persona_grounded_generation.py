from instruction_files.generator_class import GeneratorBasic

import string
import json
import random
from string import Template
import os
import settings
from collections import Counter, defaultdict
random.seed(123)
import re

instruction_dict = {
	"id": "wow",
    "Definitions": ["In this task you will be shown a conversation and persona description of a speaker. You need to generate a response to the conversation based on the conversation and the provided persona description.",
                    "Read the dialogue and the persona description text to generate a response that is grounded on the persona",
                    "Provide a response based on a dialogue and persona. The response may or may not include information from the persona."],
    "Positive Examples": [
    {
      "text": "[PERSONA] i'm 60years old. i really like to travel. i think i will retire in a few years. i am a librarian. [CONTEXT] Hello! How is your day? \ud83d\ude09 [EOT] Are you still with me? [EOT] Hi :) it was great, thanks for asking [EOT] I am not sure what that is. What else do you like ? [EOT] what about yours? [EOT] Given this context and persona, the response is:",
      "output": "i am a librarian . my retirement is so extensive",
      "context": [
        "Hello! How is your day? \ud83d\ude09",
        "Are you still with me?",
        "Hi :) it was great, thanks for asking",
        "I am not sure what that is. What else do you like ?",
        "what about yours?"
      ],
      "persona": [
        "i'm 60years old.",
        "i really like to travel.",
        "i think i will retire in a few years.",
        "i am a librarian."
      ],
      "index": 8514,
      "split": "train",
      "dataset": "convai2"
    },
    ]
}




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
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data))

            definitions = instruction_dict['Definitions']

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            for dp in datapoints:
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                context = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-self.context_max_length:])
                knowledge = dp['personality']
                if type(knowledge) is list:
                    knowledge = ' '.join(knowledge)
                if len(knowledge.split())>settings.MAX_KNOWLEDGE_LENGTH:
                    knowledge = ' '.join(knowledge.split())[:settings.MAX_KNOWLEDGE_LENGTH]
                response = dp['response']
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                if 'Searching for peer. Please wait...' in context_str or 'Partner found!' in context_str:
                    continue
                text =  settings.PERSONA_SEP + " " + knowledge +" "+ settings.CONTEXT_SEP +" "+ context_str + " " + settings.EOD_SEP
                post_prompts = [settings.QUESTION_SEP+" Given this context and provided persona, the response is",
                                settings.QUESTION_SEP+" Generate a response with the provided context and is the based on the provided persona",
                                settings.QUESTION_SEP+" Given this context generate a response which follows the provided persona",
                                settings.QUESTION_SEP+" Here is a response which is conditioned on the persona"]

                text = text +' '+ random.choice(post_prompts)
                text = re.sub(' +', ' ', text)

                output = response
                sequences.append({'text':text, 'output': output, 'metadata':{'context': dp['context'], 'persona': dp['personality']}, 'index':index, 'split':split, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
