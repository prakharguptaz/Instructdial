from instruction_files.generator_class import GeneratorBasic

import string
import json
import random
from string import Template
import os
import re
from collections import Counter, defaultdict
import settings
from utils.common import get_options_string, get_alphabetwithoptions_string, get_integerwithoptions_string


instruction_dict = {
    "id": "intent_classification",
    "Source": [
        "intent",
    ],
    "Definitions": [
                    "You will be given some dialogue text and you need to classify the intent of the text.",
                    "Given a response, classify the intent of the response.",
                    "Select the correct intent of the given text from a list of options.",
                    ],
    "Positive Examples": [
        {
            "input": "[RESPONSE] Make a reservation for 4 people for today's lunch at Daniel [ENDOFDIALOGUE]. The possible intents are: [OPTIONS] BookRestaurant||||ShareETA||||ShareCurrentLocation||||GetDirections||||GetWeather||||GetPlaceDetails||||RequestRide||||ComparePlaces||||SearchPlace||||GetTrafficInformation [QUESTION] The best option among the provided intent is",
            "outputs": [
                "BookRestaurant"
            ],
            "index": 133,
            "split": "train",
            "classes_in_options": [
                "BookRestaurant",
                "ShareETA",
                "ShareCurrentLocation",
                "GetDirections",
                "GetWeather",
                "GetPlaceDetails",
                "RequestRide",
                "ComparePlaces",
                "SearchPlace",
                "GetTrafficInformation"
            ],
            "candidates": [
                "BookRestaurant",
                "ShareETA",
                "ShareCurrentLocation",
                "GetDirections",
                "GetWeather",
                "GetPlaceDetails",
                "RequestRide",
                "ComparePlaces",
                "SearchPlace",
                "GetTrafficInformation"
            ],
            "dataset": "snips"
        },
        {
            "input": "[RESPONSE] i need to fly from washington to san francisco but i 'd like to stop over at dallas can you tell me a schedule of flights that will do that [ENDOFDIALOGUE]. The possible intents are: [OPTIONS] flight_time||||ground_fare||||ground_service||||ground_service+ground_fare||||flight||||meal||||flight_no+airline||||airfare+flight||||flight+airfare||||cheapest [QUESTION]. What is the intent for the response",
            "outputs": [
                "flight"
            ],
            "index": 1339,
            "split": "train",
            "classes_in_options": [
                "flight_time",
                "ground_fare",
                "ground_service",
                "ground_service+ground_fare",
                "flight",
                "meal",
                "flight_no+airline",
                "airfare+flight",
                "flight+airfare",
                "cheapest"
            ],
            "candidates": [
                "flight_time",
                "ground_fare",
                "ground_service",
                "ground_service+ground_fare",
                "flight",
                "meal",
                "flight_no+airline",
                "airfare+flight",
                "flight+airfare",
                "cheapest"
            ],
            "dataset": "atis"
        }
    ]
}


def list_tostringold(classes):
    assert type(classes) == list
    lenc = len(classes)
    if len(classes)<2:
        return ' '.join(classes)
    elif len(classes)==2:
        return classes[0] + ' and ' + classes[1]
    else:
        return ', '.join(classes[:-1]) + ' and ' + classes[-1]

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
            print(dataset_reader.name)
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
            # datapoints = random.sample(datapoints, min(len(datapoints), self.max_data))

            definitions = instruction_dict['Definitions']
            
            for dp in datapoints:
                definition = random.choice(definitions)
                # num_classes = random.randint(2, len(dataset_reader.intent_classes))
                # num_classes = min(num_classes, 10)
                # classes = random.sample(dataset_reader.intent_classes, num_classes)
                dp['intent_label'] = dp['intent_label'].replace('_',' ')
                # if dp['intent_label'] not in classes:
                #     classes+=[dp['intent_label']]
                classes = dataset_reader.intent_classes
                classes = [x.replace('_',' ') for x in classes]
                random.shuffle(classes)

                # single_class_selected=dp['intent_label']
                # classes_list = [str(i) for i in range(len(classes))]
                # mapped_instruction = get_integerwithoptions_string(classes)
                # answer_idx = classes.index(single_class_selected)
                # all_outputs = [str(answer_idx)]

                mapping = {'classes':get_options_string(classes)}
                instruction_sent =  f'The possible intents are: $classes'
                mapped_instruction = Template(instruction_sent).substitute(**mapping)
                all_outputs = [dp['intent_label']]
                classes_list = classes

                context = dp.get('context', '')
                if context !='' and type(context) is list:
                    context = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])

                if context!='':
                    text = settings.CONTEXT_SEP + ' ' + context + ' '+settings.RESPONSE_SEP + ' ' + dp['response']+ ' ' + settings.EOD_SEP + mapped_instruction
                else:
                    text = settings.RESPONSE_SEP + ' ' + dp['response']+ ' ' + settings.EOD_SEP + ' ' + mapped_instruction
                post_prompts = [settings.QUESTION_SEP+'. Choose the most likely intent',
                                settings.QUESTION_SEP+'. What is the intent for the response']

                text = text + ' ' + random.choice(post_prompts)
                text = re.sub(' +', ' ', text)
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                sequences.append({'input':text, 'outputs': all_outputs, 'index':index, 'split':split, 'metadata':{'intent_label':dp['intent_label'], 'context':dp.get('context')}, 'classes_in_options':classes_list, 'candidates':classes, 'dataset':dataset_reader.name})

            print('len of data', len(sequences))
        return (sequences, instruction_dict)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
