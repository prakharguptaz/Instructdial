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

instruction_dict = {
	"id": "relation_classification",
    "Source": [
        "relation",
    ],
    "Definitions": ["You will be given some conversational text and you need to predict if a provided relation is present between specified people or speakers. ",
                    "Given some conversation text, you need to find if a relation in the conversation between specified people or speakers exist or not.",
                    "Determine if a relation exists between the specified people in a conversation"],
    "Positive Examples": [
        {
            "input": "[CONTEXT] Speaker 1: Hello, Mark? Hi, it's Rachel Green. Oh no, don't you apologize. Yeah, I'll hold. He left my number at work, but he was helping his niece with her report on the pioneers. [EOT] Speaker 2: That is so made up! [EOT] Speaker 1: Yeah, oh my God, tomorrow! That, no, it's perfect. Oh God, thank you soo much. Great! Bye! I got the interview! [EOT] Speaker 3: Yay! [EOT] Speaker 2: There you go. [EOT] . Does the relation per:alternate_names exist between Speaker 1 and Rachel Green. Answer yes or no:",
            "outputs":
            [
                "yes"
            ],
            "index": 274,
            "split": "train",
            "dataset": "dialogre"
        },
        {
            "input": "[CONTEXT] Speaker 1: Hey, everybody. Happy Thanksgiving! [EOT] Speaker 2: Hey happy Thanksgiving... Pheebs! [EOT] Speaker 1: Hey, what's going on Joe? [EOT] Speaker 2: I... I... I need a good lie. [EOT] Speaker 1: Oh okay. How about the whole \"man walking on the moon\" thing. You know? You. you could. You could see the strings people! [EOT] Speaker 2: No, no, no I need a good lie to explain why I wasn't at a work thing today. [EOT] Speaker 1: Ooh, honey. You stink at lying. [EOT] Speaker 2: I do not. [EOT] Speaker 1: Oh really. Okay. let me ask you something. Yesterday at the coffee house, I went to the bathroom and when I came back, my muffin was gone-who took it? [EOT] Speaker 2: Somebody opened the door to the coffee house and a raccoon came running in, went straight for your muffin and I said \"Hey don't eat that-that's Phoebe's\" and he said... He said... \"Joey you stink at lying.\" What am I going to do? [EOT] Speaker 1: Don't worry, don't worry. We'll come up with a good lie. I'll help you practice it. [EOT] Speaker 2: Oh great, that'd be great. Thank you. [EOT] Speaker 1: Sure, what... what was the work thing? [EOT] Speaker 2: Uh... [EOT] Speaker 1: \"Pick up grandma at the airport\"? [EOT] Speaker 2: Oh... man... [EOT] . Does the relation per:alternate_names exist between Speaker 1 and Pheebs. Answer yes or no:",
            "outputs":
            [
                "yes"
            ],
            "index": 337,
            "split": "train",
            "dataset": "dialogre"
        }
    ]
}


def list_tostring(classes):
    assert type(classes) == list
    lenc = len(classes)
    if len(classes)<2:
        return ' '.join(classes)
    elif len(classes)==2:
        return classes[0] + ' and ' + classes[1]
    else:
        return ', '.join(classes[:-1]) + ', ' + classes[-1]


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
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data))

            definitions = instruction_dict['Definitions']
            yesnooptionstring = get_options_string(['yes', 'no'])
            candidates = ['yes', 'no']
            mapped_instruction = '. The possible options are: ' + yesnooptionstring

            for dp in datapoints:
                definition = random.choice(definitions)
                for rel in dp['relations']:
                    x, y = rel['x'], rel['y']
                    rels = rel['r']
                    rel_classes = rels
                    single_relation_selected = random.choice(rel_classes)
                    single_relation_selected = single_relation_selected.split(':')[-1].replace('_', ' ')
                    text = settings.CONTEXT_SEP + ' ' + (' '+settings.EOT_SEP+ ' ').join(dp['context']) + ' ' + settings.EOD_SEP + mapped_instruction

                    post_prompts = [settings.QUESTION_SEP+'. Do ' + x +' and '+ y +' have the relationship: '+ single_relation_selected+' between  them? Answer yes or no', settings.QUESTION_SEP+'. The relationship: ' + single_relation_selected+' exist between ' +x +' and '+ y+ '. Predict yes or no:', settings.QUESTION_SEP+'. The relationship: ' + single_relation_selected+' exist between ' +x +' and '+ y, settings.QUESTION_SEP+'. The relationship named: ' + single_relation_selected+' is present between ' +x +' and '+ y+ '. Predict yes or no:', settings.QUESTION_SEP+'. The relationship: ' + single_relation_selected+' is present between ' +x +' and '+ y]
                    text = text + ' ' + random.choice(post_prompts)
                    text = re.sub(' +', ' ', text)
                    output = 'yes'
                    index = dp.get('index', -1)
                    split = dp.get('split', 'unspecified')
                    sequences.append({'input':text, 'outputs': [output], 'metadata':{'context':dp['context'], 'relation':single_relation_selected, 'pair':x +'-'+ y}, 'index':index, 'candidates':candidates, 'classes_in_options':candidates, 'split':split, 'dataset':dataset_reader.name})


                    # FOr no answer, random class
                    single_relation_selected = random.choice(dataset_reader.intent_classes)
                    single_relation_selected = single_relation_selected.split(':')[-1].replace('_', ' ')
                    while single_relation_selected in rel_classes:
                        single_relation_selected = random.choice(dataset_reader.intent_classes)
                    text = settings.CONTEXT_SEP + ' ' + (' '+settings.EOT_SEP+ ' ').join(dp['context']) + ' ' + settings.EOD_SEP + mapped_instruction
                    post_prompts = [settings.QUESTION_SEP+'. Do ' + x +' and '+ y +' have the relationship: '+ single_relation_selected+' between  them? Answer yes or no', settings.QUESTION_SEP+'. The relationship: ' + single_relation_selected+' exist between ' +x +' and '+ y+ '. Predict yes or no:', settings.QUESTION_SEP+'. The relationship: ' + single_relation_selected+' exist between ' +x +' and '+ y, settings.QUESTION_SEP+'. The relationship named: ' + single_relation_selected+' is present between ' +x +' and '+ y+ '. Predict yes or no:', settings.QUESTION_SEP+'. The relationship: ' + single_relation_selected+' is present between ' +x +' and '+ y]
                    text = text + ' ' + random.choice(post_prompts)
                    text = re.sub(' +', ' ', text)
                    output = 'no'
                    index = dp.get('index', -1)
                    split = dp.get('split', 'unspecified')
                    text = re.sub(' +', ' ', text)
                    sequences.append({'input':text, 'outputs': [output], 'metadata':{'context':dp['context'], 'relation':single_relation_selected, 'pair':x +'-'+ y}, 'candidates':candidates, 'classes_in_options':candidates, 'index':index, 'split':split, 'dataset':dataset_reader.name})


        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
