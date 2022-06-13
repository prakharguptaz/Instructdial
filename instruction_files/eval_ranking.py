from instruction_files.generator_class import GeneratorBasic
from utils import extraction

import string
import json
import random
from string import Template, ascii_uppercase
import os
from collections import Counter, defaultdict
import settings
from tqdm import tqdm
import re
from utils.common import get_options_string, get_alphabetwithoptions_string

random.seed(123)

instruction_dict = {
    "id": "evaluation_ranking",
    "Definitions": [
        "Choose the best response from the provided responses to a conversation. ",
        "Given a conversation and some responses to the conversation, select the most relevant response to the conversation.",
        "Select the best response from the provided responses to a conversation",
        "Choose the best response from the provided responses to a conversation"
    ],
    "Positive Examples": [
        {
            "input": "[CONTEXT] this is @sprint great service nothing like 3g speeds in 2017 that doesn 't even work . lol <URL> [EOT] Please rank the following responses [SEP] you can send us your email address via dm . [SEP] please dm us your account information and we will be happy to assist you ",
            "output": "please dm us your account information and we will be happy to assist you . [SEP] you can send us your email address via dm .",
        }
    ]
}

post_prompts = [settings.QUESTION_SEP + " The response which is the best follow-up to the conversation is ",
                settings.QUESTION_SEP + " The correct response is ",
                settings.QUESTION_SEP + " The best response to the conversation is "]


def response_good_logic(dp, scorekey):
    score = dp['score'][scorekey]
    mid_score = (dp['score_min'] + dp['score_max'])/2
    if dp['score_max']-dp['score_min']>2: 
        if dp['score_max']-score>1:
            return None
    # else:
    #     if score < mid_score:
    #         return None

    return score

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
            split = dataset_reader.split
            
            datapoints = []
            dp = dataset_reader.get_next()
            while dp is not None:
                dp = dataset_reader.get_next()

                if dp:
                    datapoints.append(dp)
                    dp['split'] = split
            
            datapoints = random.sample(datapoints, min(len(datapoints), self.max_data))
            definitions = instruction_dict['Definitions']
            
            print(datapoints[-1])
            print(len(datapoints), 'datapoints')
            
            response_set = [d['response'] for d in datapoints]

            idx = 0
            for dp in tqdm(datapoints):
                eval_dataset_name  = dp['dataset_name']
                #we ignore dstc6 for this task
                if eval_dataset_name in ['dstc6']:
                    continue
                response = dp['response']
                if response.strip()=='' or len(response)<6:
                    continue
                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                # if 'score' in dp and 'score_min' in dp and 'score_max' in dp:
                #     mid_score = (dp['score_min'] + dp['score_max'])/2
                #     score = max(dp['score'].values())
                #     if score < mid_score:
                #         continue

                if 'score' in dp and type(dp['score']) is dict:
                    if 'overall' in dp['score']:
                        is_good = response_good_logic(dp, 'overall')
                        if is_good is None: continue
                    elif 'make_sense' in dp['score']:
                        is_good = response_good_logic(dp, 'make_sense')
                        if is_good is None: continue
                    elif 'relevance' in dp['score']:
                        is_good = response_good_logic(dp, 'relevance')
                        if is_good is None: continue
                    elif 'appropriateness' in dp['score']:
                        is_good = response_good_logic(dp, 'appropriateness')
                        if is_good is None: continue

                negative = []
                while len(negative) < 3:
                    r = random.choice(response_set)
                    if r == response or r in negative:
                        continue
                    else:
                        negative.append(r)
                answer_idx = random.randint(0, 3)
                candidates = negative[:answer_idx] + [response] + negative[answer_idx:]    
                assert candidates[answer_idx] == response

                response_str = get_alphabetwithoptions_string(candidates)
                
                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                text = f"{settings.CONTEXT_SEP} {context_str} {settings.EOD_SEP} These are the candidate responses: {response_str}"
                text = text + ' ' + random.choice(post_prompts)
                if dp['persona'] is not None:
                    text = settings.PERSONA_SEP + " " + ' '.join(dp['persona']) +" "+ text
                if dp['knowledge'] is not None:
                    text = settings.WIKI_SEP + " " + (dp['knowledge']) +" "+ text
                output = ascii_uppercase[answer_idx]

                text = re.sub(' +', ' ', text)

                candidate_options = []
                for option, candidate in zip(ascii_uppercase, candidates):
                    candidate_options.append(f'{option}')

                sequences.append({'input': text, 'outputs': output, 
                                    'index': idx, 'split': dp['split'], 'metadata': {'human_rating':dp['score'], 'eval_dataset_name':eval_dataset_name}, 
                                    'classes_in_options': candidate_options, 'candidates': candidates,
                                    'dataset': dataset_reader.name})
                idx += 1
            for s in (sequences[-3:]): print(s)

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
