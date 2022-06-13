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
    "id": "emotion_generation",
    "Source": [
        "self"
    ],
    "Definitions": [
        "In this task you will generate an utterance given a dialogue context and an emotion",
        "Using the provided emotion, generate a response to the conversation",
        "Write a response to the conversation so that the response contains the emotion providedd",
        "Generate a response to the conversation with the given emotion",
        "Create a response to the dialog using the given emotion"],
    "Positive Examples": [
        {


            "input": settings.EMOTION_SEP + " anger " +
                     settings.CONTEXT_SEP + " I won! I won! I finally won! " +
                     settings.EOT_SEP + " I won! That was my quarter! " +
                     settings.EOD_SEP + " " +
                     settings.QUESTION_SEP + " Given the context and emotion, the response is ",
            "output": "Fine! Here! Take a hike toots!",
            "index": 5987,
            "split": "train",
            "dataset": "emotionlines"
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
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data * 5))

            definitions = instruction_dict['Definitions']

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            print(len(datapoints), 'datapoints')
            dataset_sequences = []
            for dp in tqdm(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')

                context = dp['context']
                context = context[-settings.MAX_CONTEXT_NUMUTTERANCE:]
                if type(context) is str:
                    print('context should be list\n\n\n\n\n\n\n')
                    exit(0)
                if len(context) == 0:
                    context_str = ''
                else:
                    context = (' ' + settings.EOT_SEP + ' ').join(context)
                    context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                if type(dp['emotions']) is str:
                    dp['emotions'] = [dp['emotions']]
                emotion = random.choice(dp['emotions'])
                if emotion in ['neutral', 'no_emotion']:
                    continue

                output = dp['response']

                post_prompts = [settings.QUESTION_SEP + " The response with the given emotion is ",
                                settings.QUESTION_SEP + " Given the context and emotion, the response is ",
                                settings.QUESTION_SEP + " A good response using the provided emotion is ",]

                text = settings.EMOTION_SEP + " " + emotion + " " + \
                       settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP + \
                       " " + random.choice(post_prompts)

                text = re.sub(' +', ' ', text)

                dataset_sequences.append(
                    {'text': text, 'output': output, 'index': index, 'split': split, 'dataset': dataset_reader.name, 'metadata':{'emotion':emotion, 'context':dp['context']}})

            emotion_distribution = defaultdict(int)
            for dp in dataset_sequences:
                emotion = dp['metadata']['emotion']
                emotion_distribution[emotion]+=1

            num_data_toconsider = sorted(emotion_distribution.values(), reverse=True)[1]
            print(emotion_distribution, num_data_toconsider)
            selected_datapoints = []
            newemotion_distribution = defaultdict(int)
            for dp in dataset_sequences:
                emotion = dp['metadata']['emotion']
                if newemotion_distribution[emotion]<num_data_toconsider:
                    newemotion_distribution[emotion]+=1
                    selected_datapoints.append(dp)
            print(newemotion_distribution)

            sequences+=selected_datapoints

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
