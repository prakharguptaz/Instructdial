from instruction_files.generator_class import GeneratorBasic

import string
import json
import random
import copy
from string import Template
import os
import re
from collections import Counter, defaultdict
import settings

MAX_EXTRA_SLOTS = 10

instruction_dict = {
	"id": "slot_tagging",
    "Source": [
        "self",
    ],
    "Definitions": ["In this task you will be shown some dialogue utterance and you need to answer a question about the slots in the utterance",
                    "Read the dialogue utterance and predict the value of a slot. Note that the slot may be not present in the utterance.",
                    "Determine the value of the slot in the dialogue utterance. There is a chance that the slot is not present."],
    "Positive Examples": [
        {
            "text": "[CONTEXT] I need tickets to batman vs superman zipcode 20619 [ENDOFTURN] Sure! It's showing practically every hour. When would you like to see it? I can match it to a showtime. [ENDOFTURN] 8pm please [ENDOFTURN] 2 tickets [ENDOFTURN] Great! There's an 8pm at R/C Lexington Exchange Movies 12. Shall I book that for you? [ENDOFTURN] Yes please. [RESPONSE] Great! You've got 2 tickets to the 8pm showing. Enjoy the movie! [ENDOFDIALOGUE] [QUESTION]. What is the value of slot: starttime in the response",
            "index": 8740,
            "split": "train",
            "outputs": [
                "8pm"
            ],
            "dataset": "msre2e"
        },
        {
            "text": "[RESPONSE] what do you have tomorrow after 5 o'clock from atlanta to san francisco [ENDOFDIALOGUE] [QUESTION]. What is the value of slot: toloc.city_name in the response",
            "index": 1391,
            "split": "train",
            "outputs": [
                "san francisco"
            ],
            "dataset": "atis"
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


def clean_taskmaster_slots(slots_dict):
    new_dict = {}
    for k,val in slots_dict.items():
        for i, v in enumerate(val):
            if v[-1] == '.' or v[-1] == ',' or v[-1] == '?' and 'p.m' not in v.lower() and 'a.m' not in v.lower() :
                v = v[:-1]
                val[i] = v
        kwords = k.split('.')
        item_domain = kwords[0]
        item_key = '.'.join(kwords[1:3])
        if 'preference' in item_key: item_key = 'preference'
        # print(k, item_domain, item_key)
        if item_key not in domain_term_dict[item_domain]:
            new_dict[item_key] = val
            # print(new_dict[item_key], val, k)
        else:
            new_key = domain_term_dict[item_domain][item_key]
            new_dict[new_key] = val
    slots_dict = copy.deepcopy(new_dict)

    return slots_dict

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
            print('len sequences at start ', len(sequences))
            # if dataset_reader.name == 'snips':
            #     import pdb;pdb.set_trace()
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
            datapoints = random.sample(datapoints, min(len(datapoints), 20000))#self.max_data))
            definitions = instruction_dict['Definitions']
            print('len(datapoints)', len(datapoints))

            seen_slots = set()
            for dp in datapoints:
                response = dp.get('text')
                if response is None:
                    response = dp['response']
                slot = dp['slots']
                slot_dict = get_slot_values(response, slot)
                if dataset_reader.name == 'taskmaster':
                    slot_dict = clean_taskmaster_slots(slot_dict)
                seen_slots.update(slot_dict.keys())
            print(dataset_reader.name, 'num slots', len(seen_slots))


            slot_classes = set()
            for dp in datapoints:
                response = dp.get('text')
                if response is None:
                    response = dp['response']
                slot = dp['slots']
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                # import pdb;pdb.set_trace()
                slots_dict = get_slot_values(response, slot)
                if 'O' in slots_dict:
                    del slots_dict['O']
                # print(slots_dict)
                # if len(slots_dict)==0: 
                #     continue

                if dataset_reader.name == 'taskmaster':
                    slots_dict = clean_taskmaster_slots(slots_dict)

                # mapping = {}
                # mapped_definition = Template(definition).substitute(**mapping)
                # if dataset_reader.name not in ['taskmaster', 'msre2e']: # too many slots in taskmaster

                for s in seen_slots:
                    if s not in slots_dict:
                        slots_dict[s] = ['not present']
                



                if (split=='train' or split=='all') and dataset_reader.name in ['taskmaster', 'msre2e', 'atis']:
                    new_dict = {}
                    extra_slots = []
                    for k,v in slots_dict.items():
                        if v[0]=='not present':
                            extra_slots.append(k)
                        else:
                            new_dict[k] = v
                    random.shuffle(extra_slots)
                    for k in extra_slots[:min(2,len(new_dict))]:
                        new_dict[k] = ['not present']   
                    slots_dict = copy.deepcopy(new_dict)
                    # print(slots_dict)
                    # import pdb;pdb.set_trace()

                context = dp.get('context', '')
                if context !='' and type(context) is list:
                    context = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                    context = context[-settings.MAX_DIALOGUE_LENGTH:]
                if context!='':
                    text = settings.CONTEXT_SEP + ' ' + context + ' '+settings.RESPONSE_SEP + ' ' + response+ ' ' + settings.EOD_SEP
                else:
                    text = settings.RESPONSE_SEP + ' ' + response+ ' ' + settings.EOD_SEP

                for k in slots_dict:
                    v = random.choice(slots_dict[k])
                    if k == 'people' and 'restaurant' in dataset_reader.name:
                        k = 'number of people'
                    # if k == 'other':
                    #     continue
                    # if k not in slot_classes:
                    #     slot_classes.add(k)
                    # text =  settings.RESPONSE_SEP + ' ' + response +  " [EOS] Question: The value of "+ k +" mentioned in the utterance is"

                    post_prompts = [settings.QUESTION_SEP+" The value of "+ k +" mentioned in the response is", settings.QUESTION_SEP+" The value of slot "+ k +" in the response is", settings.QUESTION_SEP+" In the response, the value of slot "+ k +" is", settings.QUESTION_SEP+" What is the value of slot: "+ k +" in the response", settings.QUESTION_SEP+" Generate the value of slot "+ k +"  in the response"]
                    finaltext = text + ' ' + random.choice(post_prompts)
                    finaltext = re.sub(' +', ' ', finaltext)
                    output = v
                    sequences.append({'text':finaltext, 'index':index, 'split':split, 'outputs': [output], 'metadata':{'slot_label': k, 'response':response}, 'dataset':dataset_reader.name})

                # if len(slot_classes)>0 and random.random()<0.2:
                #     random_slot = random.choice(tuple(slot_classes))
                #     if random_slot not in slots_dict:
                #         k = random_slot
                #         post_prompts = [settings.QUESTION_SEP+" The value of "+ k +" mentioned in the response is", settings.QUESTION_SEP+" The value of slot "+ k +" in the response is", settings.QUESTION_SEP+" In the response, the value of slot "+ k +" is", settings.QUESTION_SEP+" What is the value of slot: "+ k +" in the response", settings.QUESTION_SEP+" Generate the value of slot "+ k +"  in the response"]
                #         finaltext = text + ' ' + random.choice(post_prompts)
                #         finaltext = re.sub(' +', ' ', finaltext)
                #         output = 'not present'
                #         sequences.append({'text':finaltext, 'index':index, 'split':split, 'outputs': [output], 'dataset':dataset_reader.name})
                        # print(finaltext, slots_dict)


            # if 'master' in dataset_reader.name:
            #     import pdb;pdb.set_trace()
                



        return (sequences, instruction_dict)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
