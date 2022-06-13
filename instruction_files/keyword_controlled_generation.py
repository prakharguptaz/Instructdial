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
	"id": "wow",
    "Definitions": ["In this task you will be shown a conversation context and a set of keywords. You need to generate a response to the conversation based on the context and the provided keywords.",
                    "Read the dialogue and the provided set of keywords to generate a response conditioned on the keywords.",
                    "Generate a response to a dialogue using a given set of keywords.",
                    "Use the provided keywords to generate a response to the dialogue context."],
    "Positive Examples": [
        {
            "text": "Dialogue context: Can I help you ? [EOT] Yes , I need to buy a computer for this semester . I was told it is cheaper to buy computers here . [EOT] Well , you heard right . You can get an excellent deal on a new computer here . We have great discounts for students .. Keywords to use in generation: lot of memory, desktop computer and memory",
            "output": "I don't know much about computers . But I know I want a desktop computer with a lot of memory . And I need a printer .",
            "index": 53378,
            "split": "train",
            "dataset": "dailydialog"
        },
        {
            "text": "Dialogue context: No problem ! We have a broad selection of suits , all tailored made so that it will fit perfectly . [EOT] Great ! I want a three piece suit , preferably made from Italian cashmere or wool . [EOT] Very well sir . Would you like to have some shirts made also ?. Keywords to use in generation: silver cuff link, silk ties, silver cuff, cuff link, pair of silk, ties, silver, cuff and link",
            "output": "Sure . I ' ll also take some silver cuff link and a pair of silk ties .",
            "index": 83671,
            "split": "train",
            "dataset": "dailydialog"
        },
    ]
}

def get_finalkeywords(keywords):
    random.shuffle(keywords)
    keywords = set(keywords)
    words_covered = set()
    final_keywords = []
    for k in keywords:
        is_covered = False
        for w_in_k in k.split():
            if w_in_k in words_covered:
                is_covered = True
        if not is_covered:
            final_keywords.append(k)
            words_covered |= set(k.split())
        
    return final_keywords


def list_tostring(classes):
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
    
    return ', '.join(classes)

class Generator(GeneratorBasic):
    def __init__(self, args, taskconfig, data_readers):
        self.idx = 0
        self.args = args
        self.taskconfig = taskconfig
        if 'max_data' in self.taskconfig:
            self.max_data = self.taskconfig['max_data']
        else:
            self.max_data = args.max_data
        self.context_max_length = settings.MAX_CONTEXT_NUMUTTERANCE
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
            split = dataset_reader.split
            datapoints = []
            dp = dataset_reader.get_next()
            while dp is not None:
                # print(dp)
                iterator_index+=1
                dp = dataset_reader.get_next()
                # if iterator_index>self.max_data:
                #     break
                if dp and 'index' not in dp:
                    dp['index'] = iterator_index
                if dp:
                    datapoints.append(dp)
                    dp['split'] = split
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data*2))

            definitions = instruction_dict['Definitions']

            # mapping = {}
            # mapped_definition = Template(definition).substitute(**mapping)
            print(len(datapoints), 'datapoints')
            for dp in tqdm(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')
                context = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                response = dp['response']
                keywords = extraction.get_keywords(response)
                keywords = [k[0] for k in keywords]
                if len(keywords)==0:
                    continue

                keywords = get_finalkeywords(keywords)
                #choose random number of keywords between 1 and total number of keywords detected
                chosen_keywords = random.randint(1,len(keywords))
                keywords = keywords[:chosen_keywords]
                keywords_string = list_tostring(keywords)
                # if 'personality' in dp:
                #     context =  settings.PERSONA_SEP + " " +' '.join(dp['personality'])+" Dialogue context: "+context
                #     # print(dp, response)
                # else:
                #     context = 'Dialogue context: ' + context
                # text = context + ". Keywords to use in generation: "+ keywords_string
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                if 'Searching for peer. Please wait...' in context_str or 'Partner found!' in context_str:
                    continue
                text =  settings.KEYWORDS_SEP + " " + keywords_string +" "+ settings.CONTEXT_SEP +" "+ context_str + " " + settings.EOD_SEP 
                post_prompts = [settings.QUESTION_SEP+" Given this context and keywords provided, the response is",
                                settings.QUESTION_SEP+" Generate a response with the provided context which contains the provided keywords",
                                settings.QUESTION_SEP+" Given this context generate a response which has the given keywords",
                                settings.QUESTION_SEP+" Here is a response which contains the given keywords"]
                
                text = text +' '+ random.choice(post_prompts)
                text = re.sub(' +', ' ', text)
                output = response
                sequences.append({'text':text, 'output': output, 'metadata':{'context':dp['context'], 'keywords':keywords}, 'index':index, 'split':split, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
