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
from collections import Counter, defaultdict
import settings
from utils.common import get_options_string, get_alphabetwithoptions_string, get_integerwithoptions_string

random.seed(123)

instruction_dict = {
    "id": "act_classification",
    "Definitions": [
        "In this task you will be shown a dialog context and a response. You need to determine the dialog act of the response",
        "Given a dialog context and a response, choose the dialog act from the list of dialog act options",
        "Find the dialog act from the list of options given a dialog context and its response"],
    "Positive Examples": [
        {
            "text": settings.CONTEXT_SEP + " Hi, I\'m looking for a nice German restaurant. " + settings.EOD_SEP +
                    " The possible acts are: [OPTIONS] request||||inform "
                    + settings.QUESTION_SEP + " The dialog act is ",
            "outputs": ["inform"],
            "classes_in_options": ["request", "inform"],
            "candidates": ["request", "inform"],
            "index": 5,
            "split": "train",
            "dataset": "woz"
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

            definitions = instruction_dict['Definitions']

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            print(len(datapoints), 'datapoints')
            for dp in tqdm(datapoints):
                index = dp.get('index', -1)

                # acts = dp['act_classes']
                # if type(acts) == str:
                #     acts = json.loads(acts)
                # if type(acts) == str:
                #     continue
                # if type(acts) != list:
                #     acts = list(acts.keys())
                #
                # acts = ', '.join(acts)
                # output = acts
                # if output=='':
                #     continue

                acts = dp['act_classes']
                if type(acts) is not list:
                    if type(acts) == str:
                        acts = json.loads(acts)
                    if type(acts) is dict:
                        acts = list(acts.keys())
                if len(acts)<1: continue
                assert type(acts) is list
                single_act_selected = random.choice(acts)
                classes = dataset_reader.act_classes
                set_notused = set(acts)-set([single_act_selected])
                classes = list(set(classes)-set_notused)
                classes_list = [str(i) for i in range(len(classes))]
                mapped_instruction = get_integerwithoptions_string(classes)

                answer_idx = classes.index(single_act_selected)
                output = answer_idx
                all_outputs = [str(output)]

                    # import pdb;pdb.set_trace()
                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-self.context_max_length:])
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                post_prompts = [settings.QUESTION_SEP + " The dialog act of the response is ",
                                settings.QUESTION_SEP + " The correct dialog act of the response is ",
                                settings.QUESTION_SEP + " The best option is ", ]

                text = settings.CONTEXT_SEP + " " + context_str+ ' '+settings.RESPONSE_SEP + f" {dp['response']} "  + settings.EOD_SEP \
                       +' '+ mapped_instruction \
                       + " " + random.choice(post_prompts)

                text = re.sub(' +', ' ', text)
                sequences.append(
                    {'text': text,
                     'outputs': all_outputs,
                     'index': index,
                     'split': split,
                     'metadata':{
                         'acts':acts,
                        "classes_in_options": classes_list,
                        "candidates": classes},
                     'dataset': dataset_reader.name})

        return (sequences, instruction_dict)


def __len__(self):
    return len(self.examples)


def __getitem__(self, idx):
    return self.examples[idx]
