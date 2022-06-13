from instruction_files.generator_class import GeneratorBasic

import string
import json
import random
from string import Template, ascii_uppercase
import os
import re
from collections import Counter, defaultdict
import settings
from utils.common import get_options_string, get_alphabetwithoptions_string, get_integerwithoptions_string

instruction_dict = {
	"id": "relation_classification",
    "Source": [
        "relation",
    ],
    "Definitions": ["You will be given some conversational text and you need to find a relation present between specified people or speakers. ",
                    "You will be given some conversation text and you need to find the relation in the conversation between specified people or speakers.",
                    "Determine the relation between the two speakers in the conversation."],
    "Positive Examples": [
            {
      "input": "[CONTEXT] Speaker 1: It's like this, me, no jokes. [ENDOFTURN] Speaker 2: All right, stop it, you're freaking me out. [ENDOFTURN] Speaker 3: Oh, yeah, I don't like you this way. All right, I'll see you guys later. [ENDOFTURN] Speaker 4: Bye, Richard. [ENDOFTURN] Speaker 5: Bye sweetie, I love you. [ENDOFTURN] Speaker 3: I love you, too. [ENDOFTURN] Speaker 2: I think my boyfriend ever so dreamy, I wonder what our wedding's gonna be like. [ENDOFDIALOGUE]. The possible relations are: [OPTIONS] per:schools_attended||||per:siblings||||per:parents||||per:title||||per:roommate||||per:friends||||gpe:residents_of_place||||unanswerable||||per:girl/boyfriend||||per:negative_impression [QUESTION]. The relation between Speaker 5 and sweetie is",
      "outputs": [
        "unanswerable"
      ],
      "metadata": {
        "context": [
          "Speaker 1: It's like this, me, no jokes.",
          "Speaker 2: All right, stop it, you're freaking me out.",
          "Speaker 3: Oh, yeah, I don't like you this way. All right, I'll see you guys later.",
          "Speaker 4: Bye, Richard.",
          "Speaker 5: Bye sweetie, I love you.",
          "Speaker 3: I love you, too.",
          "Speaker 2: I think my boyfriend ever so dreamy, I wonder what our wedding's gonna be like."
        ]
      },
      "index": 327,
      "split": "train",
      "classes_in_options": [
        "per:schools_attended",
        "per:siblings",
        "per:parents",
        "per:title",
        "per:roommate",
        "per:friends",
        "gpe:residents_of_place",
        "unanswerable",
        "per:girl/boyfriend",
        "per:negative_impression"
      ],
      "dataset": "dialogre"
    },
    {
      "input": "[CONTEXT] Speaker 1: We had such a great time! She\u2019s-she\u2019s incredible! I thought the-the age difference might be a problem, but it wasn\u2019t. It wasn\u2019t at all. Elizabeth is very mature for her age. A concept lost on some people! [ENDOFTURN] Speaker 2: So it\u2019s okay to date a student. [ENDOFTURN] Speaker 1: Well, not really. I mean technically it\u2019s-it\u2019s not against the rules or anything, but it is frowned upon. Especially by that professor we ran into last night, Judgey von Holierthanthou. [ENDOFTURN] Speaker 2: Well Ross, you be careful now. You don\u2019t want to get a reputation as y\u2019know Professor McNailshisstudents. [ENDOFTURN] Speaker 1: Yeah. What-what should I do? [ENDOFTURN] Speaker 3: Well Ross, it seems pretty clear. I mean what\u2019s more important? What people think or how you feel, huh? Ross, you gotta follow your heart. [ENDOFTURN] Speaker 2: Joey that is so sweet. [ENDOFDIALOGUE]. The possible relations are: [OPTIONS] per:title||||per:neighbor||||org:students||||per:works||||per:employee_or_member_of||||per:place_of_work||||per:friends||||per:alternate_names||||per:siblings||||per:acquaintance [QUESTION] The best option among the provided relations between Speaker 1 and professor is",
      "outputs": [
        "per:title"
      ],
      "metadata": {
        "context": [
          "Speaker 1: We had such a great time! She\u2019s-she\u2019s incredible! I thought the-the age difference might be a problem, but it wasn\u2019t. It wasn\u2019t at all. Elizabeth is very mature for her age. A concept lost on some people!",
          "Speaker 2: So it\u2019s okay to date a student.",
          "Speaker 1: Well, not really. I mean technically it\u2019s-it\u2019s not against the rules or anything, but it is frowned upon. Especially by that professor we ran into last night, Judgey von Holierthanthou.",
          "Speaker 2: Well Ross, you be careful now. You don\u2019t want to get a reputation as y\u2019know Professor McNailshisstudents.",
          "Speaker 1: Yeah. What-what should I do?",
          "Speaker 3: Well Ross, it seems pretty clear. I mean what\u2019s more important? What people think or how you feel, huh? Ross, you gotta follow your heart.",
          "Speaker 2: Joey that is so sweet."
        ]
      },
      "index": 1011,
      "split": "train",
      "classes_in_options": [
        "per:title",
        "per:neighbor",
        "org:students",
        "per:works",
        "per:employee_or_member_of",
        "per:place_of_work",
        "per:friends",
        "per:alternate_names",
        "per:siblings",
        "per:acquaintance"
      ],
      "dataset": "dialogre"
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
        return ', '.join(classes[:-1]) + ', ' + classes[-1]

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
        print('number of datareaders here:', len(self.data_readers))
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
            for dp in datapoints:
                definition = random.choice(definitions)
                for rel in dp['relations']:
                    x, y = rel['x'], rel['y']
                    rels = rel['r']
                    context_str = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])

                    rel_classes = rels
                    rel_classes = [x.split(':')[-1].replace('_', ' ') for x in rel_classes]
                    single_relation_selected = random.choice(rel_classes)
                    # num_classes = random.randint(2, len(dataset_reader.intent_classes))
                    # num_classes = min(num_classes, 10)
                    # classes = random.sample(dataset_reader.intent_classes, num_classes)
                    # if single_relation_selected not in classes:
                    #     classes+=[single_relation_selected]
                    classes = dataset_reader.intent_classes
                    classes = [x.split(':')[-1].replace('_', ' ') for x in classes]                    #since it is multi-label and we predict only one class, remove other gold classes from the options
                    set_notused = set(rel_classes)-set([single_relation_selected])
                    classes = list(set(classes)-set_notused)
                    classes_list = [str(i) for i in range(len(classes))]

                    random.shuffle(classes)
                    # mapping = {'classes':get_options_string(classes)}
                    # instruction_sent = '. The possible relations are: $classes'
                    # mapped_instruction = Template(instruction_sent).substitute(**mapping)
                    mapped_instruction = get_integerwithoptions_string(classes)
                    all_outputs = []
                    # for dplabel in rel_classes:
                    #     answer_idx = classes.index(dplabel)
                    #     output = answer_idx
                    #     all_outputs.append(str(output))

                    answer_idx = classes.index(single_relation_selected)
                    output = answer_idx
                    all_outputs.append(str(output))
                    text = settings.CONTEXT_SEP + ' ' + context_str + ' ' + settings.EOD_SEP + mapped_instruction
                    post_prompts = [settings.QUESTION_SEP+' The most possible relation between ' +x +' and '+ y+ ' is',
                                    settings.QUESTION_SEP+' Choose the most possible relation between ' +x +' and '+ y,
                                    settings.QUESTION_SEP+' What is the relation between ' +x +' and '+ y+ '?',
                                    settings.QUESTION_SEP+" The best option among the provided relations between " +x +' and '+ y+ ' is']

                    text = text + ' ' + random.choice(post_prompts)
                    text = re.sub(' +', ' ', text)
                    output = single_relation_selected
                    index = dp.get('index', -1)
                    split = dp.get('split', 'unspecified')
                    sequences.append({'input':text, 'outputs': all_outputs, 'metadata':{'context':dp['context'], 'relation':single_relation_selected, 'pair':x + '-'+y},  'index':index, 'split':split, 'classes_in_options':classes_list, 'candidates': classes, 'dataset':dataset_reader.name})

        return (sequences, instruction_dict)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
