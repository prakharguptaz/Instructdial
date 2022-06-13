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
from utils.common import get_options_string

random.seed(123)

instruction_dict = {
    "id": "persuasion_strategy",
    "Source": [
        "self"
    ],
    "Definitions": [
        "In this task you will be shown a dialog and you need choose a good persuasion strategy for this dialog.",
        "Select the best persuasion strategy for this dialog",
        "From a list of persuasion strategy, choose the strategy that best reflects the dialog"],
    "Positive Examples": [
        {
            "text": settings.CONTEXT_SEP + " how can i help? " + settings.EOD_SEP +
                    " The possible strategies are: [OPTIONS] request||||inform  " +
                    settings.QUESTION_SEP + " The strategy is ",
            "outputs": ["proposition-of-donation"],
            "classes_in_options": ["logical-appeal", "self-modeling", "closing", "personal-story", "confirm-donation",
                                   "off-task", "donation-information", "praise-user", "proposition-of-donation",
                                   "credibility-appeal", "acknowledgement", "positive-to-inquiry",
                                   "negative-to-inquiry", "ask-not-donate-reason", "other", "comment-partner",
                                   "emotion-appeal", "thank", "ask-donate-more", "ask-donation-amount",
                                   "foot-in-the-door", "source-related-inquiry", "you-are-welcome",
                                   "task-related-inquiry", "greeting", "personal-related-inquiry",
                                   "neutral-to-inquiry"],
            "candidates": ["logical-appeal", "self-modeling", "closing", "personal-story", "confirm-donation",
                           "off-task", "donation-information", "praise-user", "proposition-of-donation",
                           "credibility-appeal", "acknowledgement", "positive-to-inquiry",
                           "negative-to-inquiry", "ask-not-donate-reason", "other", "comment-partner",
                           "emotion-appeal", "thank", "ask-donate-more", "ask-donation-amount",
                           "foot-in-the-door", "source-related-inquiry", "you-are-welcome",
                           "task-related-inquiry", "greeting", "personal-related-inquiry",
                           "neutral-to-inquiry"],
            "index": 123,
            "dataset": "persuasion",
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

            print(len(datapoints), 'datapoints')

            for dp in tqdm(datapoints):
                output = random.choice(dp['strategy'])
                if type(dp['context']) is str:
                    dp['context'] = [dp['context']]
                    # print('converting str to list')
                    # import pdb;pdb.set_trace()
                # import pdb;pdb.set_trace()
                context = dp['context'].append(settings.RESPONSE_SEP + ' ' + dp['response'])
                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-self.context_max_length:])
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                post_prompts = [settings.QUESTION_SEP + " The strategy is ",
                                settings.QUESTION_SEP + " The correct strategy act is ",
                                settings.QUESTION_SEP + "The best option is "]

                text = settings.CONTEXT_SEP + " " + context_str + " " + settings.EOD_SEP \
                       + " The possible strategies are: " + get_options_string(dataset_reader.strategy_classes) \
                       + " " + random.choice(post_prompts)

                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                sequences.append({'input': text,
                                  'output': output,
                                  'index': index,
                                  'split': split,
                                  "classes_in_options": dataset_reader.strategy_classes,
                                  "candidates": dataset_reader.strategy_classes,
                                  'dataset': dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
