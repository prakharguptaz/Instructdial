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
import copy

instruction_dict = {
    "id": "slot_present",
    "Source": [
        "self"
    ],
    "Definitions": ["In this task you will be shown some dialogue text and you need to predict if a slot if present in the utterance.",
                    "Read the dialogue utterance and predict if a slot is present in the utterance.",
                    "Decided if a slot exists in the utterance given the dialogue text"],
    "Positive Examples": [
        {
            "text": "[RESPONSE] Yes. That sounds great. Can I scheduled a visit for the 9th of March? [ENDOFDIALOGUE]. The possible options are: [OPTIONS] yes||||no [QUESTION]. The slot visit date is present in the response?",
            "output": [
                "yes"
            ],
            "index": 2832,
            "candidates": [
                "yes",
                "no"
            ],
            "classes_in_options": [
                "yes",
                "no"
            ],
            "split": "train",
            "dataset": "slot-dstc8_sgd"
        },
        {
            "text": "[RESPONSE] time An hour before 6:30pm [ENDOFDIALOGUE]. The possible options are: [OPTIONS] yes||||no [QUESTION]. Is the slot time present in the response?",
            "output": [
                "yes"
            ],
            "index": 5418,
            "candidates": [
                "yes",
                "no"
            ],
            "classes_in_options": [
                "yes",
                "no"


            ],
            "split": "train",
            "dataset": "slot-restaurant8k"
        }
    ]
}


domain_term_dict = {}
domain_term_dict['uber_lyft'] = {'location.from':'from location',
                                'location.to':'to location',
                                'type.ride':'ride type',
                                'num.people':'number of people',
                                'price.estimate':'estimate price',
                                'duration.estimate':'estimate duration',
                                'time.pickup':'pickup time',
                                'time.dropoff':'dropoff time'}

domain_term_dict['movie_ticket'] = {'name.movie':'movie name',
                                   'name.theater':'theater name',
                                   'num.tickets':'number of tickets',
                                   'time.start':'start time',
                                   'location.theater':'theater location',
                                   'price.ticket':'ticket price',
                                   'type.screening':'screening type',
                                   'time.end':'end time',
                                   'time.duration':'duration time'}

domain_term_dict['restaurant_reservation'] = {'name.restaurant':'restaurant name',
                                             'name.reservation':'reservation name',
                                             'num.guests':'number of guests',
                                             'time.reservation':'reservation time',
                                             'type.seating':'seating type',
                                             'location.restaurant':'restaurant location'}

domain_term_dict['coffee_ordering'] = {'location.store':'store location',
                                      'name.drink':'drink name',
                                      'size.drink':'drink size',
                                      'num.drink':'number of drink',
                                      'type.milk':'milk type',
                                      'preference':'preference'}

domain_term_dict['pizza_ordering'] = {'name.store':'store name',
                                     'name.pizza':'pizza name',
                                     'size.pizza':'pizza size',
                                     'type.topping':'topping type',
                                     'type.crust':'crust type',
                                     'preference':'preference',
                                     'location.store':'store location'}

domain_term_dict['auto_repair'] = {'name.store':'store name',
                                  'name.customer':'customer name',
                                  'date.appt':'appt date',
                                  'time.appt':'appt time',
                                  'reason.appt':'appt reason',
                                  'name.vehicle':'vehicle name',
                                  'year.vehicle':'vehicle year',
                                  'location.store':'store location'}


# def clean_taskmaster_slots(slots_dict):
#     new_dict = {}
#     for k,val in slots_dict.items():
#         for i, v in enumerate(val):
#             if v[-1] == '.' or v[-1] == ',' or v[-1] == '?' and 'p.m' not in v.lower() and 'a.m' not in v.lower() :
#                 v = v[:-1]
#                 val[i] = v
#         kwords = k.split('.')
#         item_domain = kwords[0]
#         item_key = '.'.join(kwords[1:3])
#         if 'preference' in item_key: item_key = 'preference'
#         # print(k, item_domain, item_key)
#         if item_key not in domain_term_dict[item_domain]:
#             new_dict[item_key] = val
#             # print(new_dict[item_key], val, k)
#         else:
#             new_key = domain_term_dict[item_domain][item_key]
#             new_dict[new_key] = val
#     slots_dict = copy.deepcopy(new_dict)
#
#     return slots_dict

def clean_taskmaster_slots(slots_dict):
    for k,val in slots_dict.items():
        for i, v in enumerate(val):
            if v[-1] == '.' or v[-1] == ',' or v[-1] == '?' and 'p.m' not in v.lower() and 'a.m' not in v.lower() :
                v = v[:-1]
                val[i] = v
        slots_dict[k] = val

    return slots_dict

def clean_taskmaster_slotname(slotname):
    kwords = slotname.split('.')
    item_domain = kwords[0]
    item_key = '.'.join(kwords[1:3])
    if 'preference' in item_key: item_key = 'preference'
    new_slotname = domain_term_dict[item_domain].get(item_key, item_key)
    item_domain_name = ' '.join(item_domain.split('_'))
    new_slotname = new_slotname.replace('_', ' ')
    # slot_text = f"{new_slotname} in domain {item_domain_name}"
    slot_text = f"{new_slotname}"

    return slot_text


def get_slot_values(text, slot):
    text_words = text.split()
    slot_words = slot.split()
    slot_inter = []
    word_inter = []
    slots = defaultdict(list)
    for i, (t,s) in enumerate(zip(text_words,slot_words)):
        if "B-" in s:
            if len(slot_inter)>0:
                slot_val = (slot_inter)
                word_val = ' '.join(word_inter)
                # print('word_valb', word_val)
                slots[slot_val[0].replace('I-', '').replace('B-', '')].append(word_val)
                # print(slot_val, word_val)
            slot_inter, word_inter = [], []
            slot_inter.append(s)
            word_inter.append(t)
        elif 'I-' in s:
            slot_inter.append(s)
            word_inter.append(t)            
        elif s=='O':
            if len(slot_inter)>0:
                slot_val = (slot_inter)
                word_val = ' '.join(word_inter)
                # print('word_val', word_val)
                slots[slot_val[0].replace('I-', '').replace('B-', '')].append(word_val)
                # print(slot_val, word_val)
            slot_inter, word_inter = [], []
    if len(slot_inter)>0:
        slot_val = (slot_inter)
        word_val = ' '.join(word_inter)
        slots[slot_val[0].replace('I-', '').replace('B-', '')].append(word_val)
        # print(slot_val, word_val)

    return slots





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
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data))
            definitions = instruction_dict['Definitions']
            yesnooptionstring = get_options_string(['yes', 'no'])
            candidates = ['yes', 'no']
            mapped_instruction = '. The possible options are: ' + yesnooptionstring

            print('len(datapoints)', len(datapoints))
            slot_classes = (dataset_reader.slot_classes)
            slot_classes = [x.replace('I-', '').replace('B-', '') for x in slot_classes]
            slot_classes = list(set(slot_classes) - set(['[PAD]', 'O']))

            if dataset_reader.name=='taskmaster':
                new_slot_classnames = []
                for k in slot_classes:
                    newslotname = clean_taskmaster_slotname(k)
                    new_slot_classnames.append(newslotname)
                slot_classes = new_slot_classnames[:]
                # print(slot_classes)

            for dp in datapoints:
                slot =  dp['slots']
                response = dp.get('text')
                if response is None:
                    response = dp['response']
                slots_dict = get_slot_values(response, slot)
                if dataset_reader.name == 'taskmaster':
                    new_slots_dict = {}
                    for k,v in slots_dict.items():
                        newk = clean_taskmaster_slotname(k)
                        new_slots_dict[newk] = v
                    slots_dict = copy.deepcopy(new_slots_dict)
                    # slots_dict = clean_taskmaster_slots(slots_dict)
                if 'O' in slots_dict:
                    del slots_dict['O']
                # print(slots_dict)
                # if len(slots_dict)==0:
                #     continue
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                # mapping = {}
                # mapped_definition = Template(definition).substitute(**mapping)
                for k in slots_dict:
                    # v = random.choice(slots_dict[k])
                    kval = k.replace('_', ' ').replace('.', ' ')
                    slot = kval
                    # text = settings.RESPONSE_SEP + ' ' + response+ " " + settings.EOT_SEP+" Question: Is the slot "+ slot +" present in the utterance?"
                    context = dp.get('context', '')
                    if context !='' and type(context) is list:
                        context = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                    if context!='':
                        text = settings.CONTEXT_SEP + ' ' + context + ' '+settings.RESPONSE_SEP + ' ' + response+ ' ' + settings.EOD_SEP + mapped_instruction
                    else:
                        text = settings.RESPONSE_SEP + ' ' + response+ ' ' + settings.EOD_SEP + mapped_instruction
                    if 'auto_repair' in kval:
                        print(response)
                        print(text)
                        import pdb; pdb.set_trace()
                    post_prompts = [settings.QUESTION_SEP+'. Is the slot: ' + slot + ' present in the response?', settings.QUESTION_SEP+'. The slot: ' + slot + ' is present in the response?', settings.QUESTION_SEP+'. Does the provided slot exist the response? The slot is: ' + slot, settings.QUESTION_SEP+'. Does the provided slot exist the response? The slot is: ' + slot, settings.QUESTION_SEP+'. Is the slot: ' + slot + ' in the response?']

                    text = text + ' ' + random.choice(post_prompts)
                    text = re.sub(' +', ' ', text)
                    output = 'yes'
                    sequences.append({'text':text, 'output': [output], 'index':index, 'candidates':candidates, 'classes_in_options':candidates, 'metadata':{'slot_label': slot, 'response':response}, 'split':split, 'dataset':dataset_reader.name})
                    random_slot = None
                    while True:
                        random_slot = random.choice(slot_classes)
                        if random_slot.replace(' ', '-') not in slots_dict:
                            break
                    slot = random_slot
                    slot = slot.replace('_', ' ').replace('.', ' ')
                    # text = settings.RESPONSE_SEP + ' ' + response+ " " + settings.EOT_SEP +"Question: Is the slot "+ slot +" present in the utterance?"
                    post_prompts = [settings.QUESTION_SEP+'. Is the slot: ' + slot + ' present in the response?', settings.QUESTION_SEP+'. The slot: ' + slot + ' is present in the response?', settings.QUESTION_SEP+'. Does the provided slot exist the response? The slot is: ' + slot, settings.QUESTION_SEP+'. Does the provided slot exist the response? The slot is: ' + slot, settings.QUESTION_SEP+'. Is the slot: ' + slot + ' in the response?']
                    if context!='':
                        text = settings.CONTEXT_SEP + ' ' + context + ' '+settings.RESPONSE_SEP + ' ' + response+ ' ' + settings.EOD_SEP + mapped_instruction
                    else:
                        text = settings.RESPONSE_SEP + ' ' + response+ ' ' + settings.EOD_SEP + mapped_instruction

                    text = text + ' ' + random.choice(post_prompts)
                    text = re.sub(' +', ' ', text)
                    output = 'no'
                    sequences.append({'input':text, 'outputs': [output], 'index':index, 'candidates':candidates, 'classes_in_options':candidates, 'metadata':{'slot_label': slot, 'response':response}, 'split':split, 'dataset':dataset_reader.name})
            print(f'after {dataset_reader.name} size', len(sequences))
        return (sequences, instruction_dict)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
