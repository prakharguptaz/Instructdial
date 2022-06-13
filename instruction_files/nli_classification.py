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
	"id": "intent_classification",
    "Source": [
        "intent",
    ],
    "Definitions": ["You will be given some dialogue text and a response and you need to find the class of relation between the response and the context.",
                    "Given a conversation response and the context, you need to classify the relation between them among the provided classes.",
                    "Choose the relation between a given context and response from a list of options"],
    "Positive Examples": [
            {
      "input": "[CONTEXT] i am named after a cartoon fox . [RESPONSE] i have a dog . [ENDOFDIALOGUE]. The possible classes are: positive||||negative||||neutral [QUESTION]. The predicted class based on the context and response is",
      "outputs": [
        "neutral"
      ],
      "metadata": {
        "context": [
          "i am named after a cartoon fox ."
        ]
      },
      "index": 123306,
      "split": "train",
      "classes_in_options": [
        "negative",
        "positive",
        "neutral"
      ],
      "candidates": [
        "negative",
        "positive",
        "neutral"
      ],
      "dataset": "dnli"
    },
    {
      "input": "[CONTEXT] what is up , party person ? [ENDOFTURN] nothing much just enjoying a day off work what about you [ENDOFTURN] oh , just waiting for my toenails to dry . [ENDOFTURN] lol are they fast drying ? [ENDOFTURN] Kind of slow lol. [RESPONSE] I know they dry fast. [ENDOFDIALOGUE]. The possible classes are: uncontradicted||||contradicted [QUESTION]. What is the class given the context and the response",
      "outputs": [
        "contradict"
      ],
      "metadata": {
        "context": [
          "what is up , party person ?",
          "nothing much just enjoying a day off work what about you",
          "oh , just waiting for my toenails to dry .",
          "lol are they fast drying ?",
          "Kind of slow lol."
        ]
      },
      "index": 10063,
      "split": "train",
      "classes_in_options": [
        "uncontradicted",
        "contradicted"
      ],
      "candidates": [
        "uncontradicted",
        "contradicted"
      ],
      "dataset": "decode"
    }
    ]
}


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
                classes = dataset_reader.labels
                random.shuffle(classes)
                mapping = {'classes':list_tostring(classes)}
                instruction_sent = f'. The possible classes are: {settings.OPTION_TOKEN} $classes'
                mapped_instruction = Template(instruction_sent).substitute(**mapping)

                response_string, context_string = None, None
                context_list = []
                if 'sentence2' in dp:
                    context_string = dp['sentence1']
                    response_string = dp['sentence2']
                    context_list = [context_string]
                else:
                    context_string = (' '+settings.EOT_SEP+ ' ').join(dp['context'])
                    response_string = dp['response']
                    context_list = dp['context']

                # if dataset_reader.name == ''dnli
                text = settings.CONTEXT_SEP+' ' +context_string +' '+settings.RESPONSE_SEP+' ' + response_string +' ' + settings.EOD_SEP + ' '+ mapped_instruction
                post_prompts = [settings.QUESTION_SEP+'. The predicted class based on the context and response is',
                                settings.QUESTION_SEP+'. Choose the most possible class',
                                settings.QUESTION_SEP+'. What is the class given the context and the response',
                                settings.QUESTION_SEP+" The best option among the provided classes is"]

                text = text + ' ' + random.choice(post_prompts)
                output = dp['label']
                text = re.sub(' +', ' ', text)
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                sequences.append({'input':text, 'outputs': [output], 'metadata':{'context': context_list}, 'index':index, 'split':split, 'classes_in_options':classes, 'candidates': classes, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
