from instruction_files.generator_class import GeneratorBasic

import string
import json
import random
from string import Template
import os
import re
from collections import Counter, defaultdict
import settings


instruction_dict = {
	"id": "slot_tagging",
    "Source": [
        "self",
    ],
    "Definitions": ["In this task you will be shown some dialogue utterance and you need to answer a question about the slots in the utterance",
                    "Read the dialogue utterance and predict the value of a slot.",
                    "Determine the value of the slot in the dialogue utterance."],
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
            print('len(datapoints)', len(datapoints))

            seen_slots = set()
            for dp in datapoints:
                response = dp.get('text')
                if response is None:
                    response = dp['response']
                slot = dp['slots']
                slot_dict = get_slot_values(response, slot)
                seen_slots.update(slot_dict.keys())
            print(dataset_reader.name, 'num slots', len(seen_slots))
            # import pdb;pdb.set_trace()            
            for i, dp in enumerate(datapoints):
                response = dp.get('text')
                if response is None:
                    response = dp['response']
                slot = dp['slots']
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                if 'slot_to_word' in dp:
                    slots_dict = dp['slot_to_word']
                else:
                    slots_dict = get_slot_values(response, slot)
                if 'O' in slots_dict:
                    del slots_dict['O']
                # print(slots_dict)
                # if len(slots_dict)==0: 
                #     continue
                # mapping = {}
                # mapped_definition = Template(definition).substitute(**mapping)
                # print(response, slots_dict)
                # import pdb;pdb.set_trace()
                for s in seen_slots:
                    if s not in slots_dict:
                        slots_dict[s] = ['not present']

                for k in slots_dict:
                    v = random.choice(slots_dict[k])
                    # text =  settings.RESPONSE_SEP + ' ' + response +  " [EOS] Question: The value of "+ k +" mentioned in the utterance is"
                    context = dp.get('context', '')
                    if context !='' and type(context) is list:
                        context = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                    if context!='':
                        text = settings.CONTEXT_SEP + ' ' + context + ' '+settings.RESPONSE_SEP + ' ' + response+ ' ' + settings.EOD_SEP
                    else:
                        text = settings.RESPONSE_SEP + ' ' + response+ ' ' + settings.EOD_SEP
                    post_prompts = [settings.QUESTION_SEP+". The value of "+ k +" mentioned in the response is", settings.QUESTION_SEP+". The value of slot "+ k +" in the response is", settings.QUESTION_SEP+". In the response, the value of slot "+ k +" is", settings.QUESTION_SEP+". What is the value of slot: "+ k +" in the response", settings.QUESTION_SEP+". Generate the value of slot "+ k +"  in the response is"]

                    text = text + ' ' + random.choice(post_prompts)
                    text = re.sub(' +', ' ', text)
                    output = v
                    sequences.append({'text':text, 'index':index, 'split':split, 'outputs': [output], 'dataset':dataset_reader.name, 'metadata':{'slot_label': k, 'domain':dp['domain'], 'response':response}})
                # sequences.append({'input':text, 'outputs': [output], 'index':index, 'split':split, 'dataset':dataset_reader.name})
                # print(i, len(sequences))
            print(dataset_reader.name, len(sequences))

        print(sequences[-1])
        import pdb;pdb.set_trace()

        return (sequences, instruction_dict)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
