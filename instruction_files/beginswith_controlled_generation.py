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
    "Definitions": [
        "In this task you will be shown a conversation context and an initial phrase. You need to generate a response to the conversation based on the context which starts with the provided initial phrase.",
        "Read the dialogue and generate a response which starts with the provided initial phrase.",
        "In this task you are given a dialog and an initial phrase. You need to generate a response which begins with the initial phrase",
        "Given an initial phrase and a conversation, generate a response with begins with the provided initial phrase"],
    "Positive Examples": [
        {
            "text": "[INITIAL PHRASE] I tell ya [CONTEXT] Okay , can I ask you something direct ? [ENDOFTURN] Ha ! It's not like you've ever been one to beat around the bush . [ENDOFTURN] Fair enough . Give it to me straight . Did she bully you into this ? [ENDOFTURN] No , seriously . . . I really want this . [ENDOFDIALOGUE] [QUESTION] Given this context generate a response which starts with the given initial sentence, the response is",
            "output": "I tell ya , when I got the invite , it really threw me for a loop . You've done a complete 180 .",
            "index": 865,
            "metadata": {
                "context": [
                    "Okay , can I ask you something direct ?",
                    "Ha ! It's not like you've ever been one to beat around the bush .",
                    "Fair enough . Give it to me straight . Did she bully you into this ?",
                    "No , seriously . . . I really want this ."
                ],
                "beginswith": "I tell ya"
            },
            "split": "train",
            "dataset": "dailydialog"
        },
        {
            "text": "[INITIAL PHRASE] May I have [CONTEXT] Good morning , Maintenance Department . [ENDOFTURN] Hello . I'm having a problem with my air conditioner . [ENDOFTURN] Which air conditioner ? [ENDOFTURN] The one in the bedroom . [ENDOFTURN] What seems to be the problem ? [ENDOFTURN] There's no cold air coming out . [ENDOFDIALOGUE] [QUESTION] Here is a response which starts with the given initial phrase",
            "output": "May I have your room number , please ?",
            "index": 41348,
            "metadata": {
                "context": [
                    "Good morning , Maintenance Department .",
                    "Hello . I'm having a problem with my air conditioner .",
                    "Which air conditioner ?",
                    "The one in the bedroom .",
                    "What seems to be the problem ?",
                    "There's no cold air coming out ."
                ],
                "beginswith": "May I have"
            },
            "split": "train",
            "dataset": "dailydialog"
        }
    ]
}


def get_startphrase(response):
    response_words = response.split()
    max_startwords = min(5, int(len(response_words) / 2))
    if max_startwords < 1:
        return ''
    num_wordsinphrase = random.randint(1, max_startwords)

    return ' '.join(response_words[:num_wordsinphrase])


def list_tostring(classes):
    assert type(classes) == list
    lenc = len(classes)
    if len(classes) < 2:
        return ' '.join(classes)
    elif len(classes) == 2:
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
            print(dataset_reader.name)
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
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data * 5))

            definitions = instruction_dict['Definitions']

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            print(len(datapoints), 'datapoints')
            for dp in tqdm(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                # print(dp.keys())
                # import pdb;pdb.set_trace()
                if type(dp['context']) is str:
                    dp['context'] = [dp['context']]
                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                response = dp['response']
                phrase = get_startphrase(response)
                # text = "Phrase to begin the response with: "+ phrase
                # if 'personality' in dp:
                #     context =  settings.PERSONA_SEP + " " +' '.join(dp['personality'])+" Dialogue context: "+context
                # else:
                #     context = '[dialogue context]: ' + context

                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                text = settings.INITIALPHRASE_SEP + " " + phrase + " " + settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP
                post_prompts = [settings.QUESTION_SEP + " Given this context and initial phrase, the response is",
                                settings.QUESTION_SEP + " Generate a response with the provided context which begins with the provided initial phrase",
                                settings.QUESTION_SEP + " Given this context generate a response which starts with the given initial sentence",
                                settings.QUESTION_SEP + " Here is a response which starts with the given initial phrase"]

                text = text + ' ' + random.choice(post_prompts)
                text = re.sub(' +', ' ', text)
                output = response
                sequences.append({'text': text, 'output': output, 'index': index,
                                  'metadata': {'context': dp['context'], 'beginswith': phrase}, 'split': split,
                                  'dataset': dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
