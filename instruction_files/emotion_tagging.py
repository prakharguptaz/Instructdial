from instruction_files.generator_class import GeneratorBasic
from utils import extraction

import string
import json
import random
from string import Template
import os
from tqdm import tqdm
import re
from collections import Counter, defaultdict
import settings
from utils.common import get_options_string, get_alphabetwithoptions_string

random.seed(123)

instruction_dict = {
    "id": "emotion_tagging",
    "Source": [
        "self"
    ],
    "Definitions": [
        "In this task you will be shown some dialogue utterance and you need predict the emotions in represented by the dialogue utterance",
        "Given a text, choose the correct emotion contained in the response from a list of options",
        "Given a dilogue context and a response, choose the correct emotion of the response from a list of options",
        "Select the emotion represented in the given response"],
    "Positive Examples": [
        {
            "input": settings.CONTEXT_SEP + " Hi Tag! Hey, so did you have fun with uh, with Joey last night? " +
                     settings.EOT_SEP + " Oh yeah! We went to the Knicks game. " +
                     settings.EOD_SEP + " The possible emotions are: [OPTIONS] disgust||||fear||||neutral||||anger||||sadness||||joy||||surprise " +
                     settings.QUESTION_SEP + " The emotions in the dialog are ",
            "outputs": ["joy"],
            "index": 4150,
            "split": "train",
            "classes_in_options": [
                "disgust",
                "fear",
                "neutral",
                "anger",
                "sadness",
                "joy",
                "surprise"
            ],
            "candidates": [
                "disgust",
                "fear",
                "neutral",
                "anger",
                "sadness",
                "joy",
                "surprise"
            ],
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

        num_neutral = 0
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

                output = dp['emotions']

                post_prompts = [settings.QUESTION_SEP + " The emotion in the response is ",
                                settings.QUESTION_SEP + " The correct emotion of the response is ",
                                settings.QUESTION_SEP + " The emotion that best describe the response is "]


                text = f"{settings.CONTEXT_SEP} {context_str} {settings.RESPONSE_SEP} {dp['response']} {settings.EOD_SEP}" + \
                       " The possible emotions are: " + get_options_string(dataset_reader.emotion_classes) + \
                       " " + random.choice(post_prompts)

                text = re.sub(' +', ' ', text)
                # print(dataset_reader.emotion_classes)
                # print(output)
                if type(output) ==str:
                    output = [output]
                if 'neutral' in output or 'no_emotion' in output:
                    num_neutral+=1
                # import pdb;pdb.set_trace()
                dataset_sequences.append(
                    {'text': text,
                     'output': output,
                     'index': index,
                     'split': split,
                     'metadata':{'context':dp['context'], 'response':dp['response']},
                     "classes_in_options": dataset_reader.emotion_classes,
                     "candidates": dataset_reader.emotion_classes,
                     'dataset': dataset_reader.name})


            # we will subsample dataset since emotion tags are skewed. We will use the count of the second most freq tag to limit
            emotion_distribution = defaultdict(int)
            for dp in dataset_sequences:
                emotions = dp['output']
                for emotion in emotions:
                    emotion_distribution[emotion]+=1

            num_data_toconsider = sorted(emotion_distribution.values(), reverse=True)[1]
            print(emotion_distribution, num_data_toconsider)
            selected_datapoints = []
            newemotion_distribution = defaultdict(int)
            for dp in dataset_sequences:
                emotions = dp['output']
                for emotion in emotions:
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
