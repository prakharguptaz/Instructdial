from instruction_files.generator_class import GeneratorBasic

import string
import json
import random
from string import Template
import os
import settings
from collections import Counter, defaultdict
from tqdm import tqdm
import re

instruction_dict = {
    "id": "db-based-generation",
    "Definitions": [
        "Read the dialogue, and suggested action and information to generate a response",
        "Use the suggested action and information to generate a response",
        "Generate a response to the conversation using the provided action and information",
    ],
    "Positive Examples": [
        {"text":
             settings.STATE_SEP + " {\"hotel-parking\": \"yes\", \"hotel-type\": \"guest house\"} " +
             settings.DB_SEP + " {\"Type\": \"guesthouse \", \"Stars\": \"1\"} " +
             settings.CONTEXT_SEP + " there are 21 guesthouses which offer free parking . which area do you prefer to stay in ? " +
             settings.EOT_SEP + " i am open to any area , but the hotel should definitely have only 1 star . " +
             settings.EOD_SEP + " " +
             settings.QUESTION_SEP + " Given this context, db, and state, the response is ",
         "output": "i was unable to find a 1 star guesthouse with free parking . would you like to up the stars ?",
         "index": 5714,
         "split": "train",
         "dataset": "multiwoz"
         }
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
            dataset_reader.idx = 0
            iterator_index = 0
            split = dataset_reader.split
            datapoints = []
            dp = dataset_reader.get_next()
            while dp is not None:
                # print(dp)
                iterator_index += 1
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
            for i, dp in tqdm(enumerate(datapoints)):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')

                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-self.context_max_length:])
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                post_prompts = [settings.QUESTION_SEP + " Given the information about dialog the response is ",
                                settings.QUESTION_SEP + " The response is "]
                # text = settings.STATE_SEP + " " + dp['state'] + " " + \
                #        settings.DB_SEP + " " + dp['db'] + " " + \
                #        settings.CONTEXT_SEP + " " + context_str + " " + settings.EOT_SEP + " " + \
                #        random.choice(post_prompts)
                action_string = ''
                act_obj = json.loads(dp['acts'])
                if type(act_obj) is str:
                    action_string = act_obj.lower()
                elif type(act_obj) is dict:
                    for k in act_obj.keys():
                        action_string +=  f' Action {k} with following details - '
                        for detailitem in act_obj[k]:
                            if detailitem[0]=="none":
                                action_string+='no detail'
                            else:
                                action_string += detailitem[0] + ' is '+ detailitem[1] + ', '
                        action_string+='.'
                else:
                    action_string = 'no annotation'
                action_string = action_string.replace(', .', '.')
 
                text = settings.ACT_SEP  + action_string+ " " +   settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP + " " + \
                       random.choice(post_prompts)

                output = dp['response']
                text = re.sub(' +', ' ', text)

                sequences.append({'text': text,
                                  'output': output,
                                  'metadata':{'context':dp['context'], 'action':dp['acts'], 'sys_act':list(dp['sys_act'])},
                                  'index': index,
                                  'split': split,
                                  'dataset': dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
