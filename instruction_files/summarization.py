from instruction_files.generator_class import GeneratorBasic

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
    "id": "summarization",
    "Source": [
        "self"
    ],
    "Definitions": [
        "In this task you will be shown a conversation context and you need to generate a summary of the conversation.",
        "Return a summarization of the provided conversation.",
        "Summarize the given conversation context and return it."],
    "Positive Examples": [
        {
            "input": "Generate a summary for the following dialog context. [CONTEXT] #Person2#: OK. [EOT] #Person1#: Well, how old are you? [EOT] #Person2#: 16. [EOT] #Person1#: Right. When you leave school do you think you'll get your own home away from your parents? [EOT] #Person2#: Oh yes, I'm sure I will. [EOT] #Person1#: Do you think you'll get married in the next 5 years say? [EOT] #Person2#: Probably but I certainly don't want children yet. I'm too young. [EOT] #Person1#: OK, would you like to travel? [EOT] #Person2#: I'd like to. But I don't think I will although I have a lot of time. Anyway, you certainly need money for that. [EOT] #Person1#: Thanks very much. [EOT] [SUMMARY] Given this dialog context, the summary for this dialog is: ",
            "output": "#Person1# asks #Person2#, who is 16, some questions about future home, marriage, and traveling.",
            "index": 2974,
            "split": "train",
            "dataset": "dialogsum"

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
        if 'context_max_length' in self.taskconfig:
            self.context_max_length = self.taskconfig['context_max_length']
        else:
            self.context_max_length = 3
        self.output_folder = args.tasks_output_folder
        self.out_file = self.taskconfig['task_name']
        self.data_readers = data_readers
        self.examples = []

    def generate_data(self):
        print('Number of datareaders:', len(self.data_readers))
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

            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data * 5))

            definitions = instruction_dict['Definitions']
            print(len(datapoints), 'datapoints')

            post_prompts = ["Given this dialogue context, the summary for this dialogue is: ",
                            "For this dialogue, the summary is: ",
                            "Given this dialogue context, its summary is the following: ",
                            "The summary for this dialogue is: ",
                            "Generate a summary for this dialogue",]

            for dp in tqdm(datapoints):
                if type(dp['context']) is str:
                    context = (' '+settings.EOT_SEP+' ').join(dp['context'].split("\n")) + " " + settings.EOD_SEP
                else:
                    context = (' '+settings.EOT_SEP+' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:]) + " " + settings.EOD_SEP
                # Truncate context to necessary length
                context = " ".join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                post_prompt = random.choice(post_prompts)
                text = "{} {} {} {}".format(settings.CONTEXT_SEP, context, settings.QUESTION_SEP, post_prompt)

                output = dp['summary']
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                sequences.append({'input': text,
                                  'output': output,
                                  'index': index,
                                  'split': split,
                                  'dataset': dataset_reader.name})

        return sequences, instruction_dict

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
