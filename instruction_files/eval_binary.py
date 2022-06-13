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
from utils.common import get_options_string, get_alphabetwithoptions_string

random.seed(123)

instruction_dict = {
    "id": "evaluation_binary",
    "Definitions": [
        "In this task you will be shown a dialogue context followed by a response. You need to predict if the response is a good follow-up to the context. ",
        "You will be given a dialogue context and a response. Choose if the response is a contextual response to the context. ",
        "Given a conversation and a response to the conversation, select if the response is a good response to the context.",
        "Given a conversation and a response, choose if the response is a good response to the context?",
        "Select if the provided response follow the provided conversation",
        "Select if the provided response is a good response to the provided dialogue context."
    ],
    "Positive Examples": [
        {
            "input": "[CONTEXT] to holden my dad use to work for u i just want to say thank u for him having a job u r the best people ever - alainaoatley [EOT] Which is the best response? [SEP] you can send us your email address via dm . [SEP] thanks for the kind words , <USER> !",
            "output": "thanks for the kind words , <USER> ",
        }
    ]
}

post_prompts = [settings.QUESTION_SEP + " Is the response a good response to the provided dialogue context? ",
                settings.QUESTION_SEP + " Select the right option: is the response a good follow-up response? ",
                settings.QUESTION_SEP + " Is the response contextual? "]


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

def response_bad_logic(dp, scorekey):
    score = dp['score'][scorekey]
    mid_score = (dp['score_min'] + dp['score_max'])/2
    if dp['score_max']-dp['score_min']>2:
        if score-dp['score_min']>1:
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

        yesnooptionstring = get_options_string(['yes', 'no'])
        candidates = ['yes', 'no']

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

            print(datapoints[0])
            print(len(datapoints), 'datapoints')
            response_set = [dp['response'] for dp in datapoints]

            idx = 0
            print('generating data for num dp', len(datapoints))
            labelcount = {'yes':0, 'no':0, 'ambiguous':0}
            for dp in tqdm(datapoints):
                context = (' ' + settings.EOT_SEP + ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                response = dp['response']
                if response.strip()=='' or len(response)<6:
                    continue
                # if we have score in dp, then we use it to check if the response is a good one
                if 'score_min' in dp and 'score_max' in dp:
                    mid_score = (dp['score_min'] + dp['score_max'])/2
                else:
                    import pdb;pdb.set_trace()
                eval_dataset_name  = dp['dataset_name']
                # import pdb;pdb.set_trace()
                output = 'yes'
                # if 'score' in dp and type(dp['score']) is float:
                #     mid_score = (dp['score_min'] + dp['score_max'])/2
                #     score = dp['score']
                #     # import pdb;pdb.set_trace()
                #     if score < mid_score:
                #         continue
                if 'score' in dp and type(dp['score']) is dict:
                    is_good = None
                    if 'overall' in dp['score']:
                        is_good = response_good_logic(dp, 'overall')
                    elif 'turing' in dp['score']:
                        is_good = response_good_logic(dp, 'turing')
                    elif 'relevance' in dp['score']:
                        is_good = response_good_logic(dp, 'relevance')
                    elif 'appropriateness' in dp['score']:
                        is_good = response_good_logic(dp, 'appropriateness')
                    # else:

                    #     import pdb;pdb.set_trace()
                    ## now check if the response is bad, else continue
                    if is_good is None:
                        # continue
                        is_bad = None
                        if 'overall' in dp['score']:
                            is_bad = response_bad_logic(dp, 'overall')
                        elif 'make_sense' in dp['score']:
                            is_bad = response_bad_logic(dp, 'make_sense')
                        elif 'relevance' in dp['score']:
                            is_bad = response_bad_logic(dp, 'relevance')
                        elif 'appropriateness' in dp['score']:
                            is_bad = response_bad_logic(dp, 'appropriateness')

                        if is_bad is None: # response is neither good nor bad
                            output = 'ambiguous'
                        else:
                            output = 'no'
                else:
                    print('weird case')
                    import pdb; pdb.set_trace()
                    # print(is_good)
                # print(dp)
                # import pdb;pdb.set_trace()

                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])

                text = f"{settings.CONTEXT_SEP} {context_str} {settings.RESPONSE_SEP} {response} {settings.EOD_SEP} Answer Choices: {yesnooptionstring}"
                if dp['persona'] is not None:
                    text = settings.PERSONA_SEP + " " + ' '.join(dp['persona']) +" "+ text
                if dp['knowledge'] is not None:
                    text = settings.WIKI_SEP + " " + (dp['knowledge']) +" "+ text


                text = re.sub(' +', ' ', text)
                post_prompt_selected = random.choice(post_prompts)
                text = text + ' ' + post_prompt_selected #+ f' {response}'
                # output = 'yes'
                labelcount[output]+=1

                sequences.append({'input': text, 'outputs': output,
                                    'index': idx, 'split': dp['split'], 'metadata': {'human_rating':dp['score'], 'eval_dataset_name':eval_dataset_name},
                                    'candidates': candidates, 'classes_in_options': candidates,
                                    'dataset': dataset_reader.name})
                idx += 1

                # negative = random.choice(response_set)
                # while negative == response:
                #     negative = random.choice(response_set)
                # text = f"{settings.CONTEXT_SEP} {context_str} {settings.RESPONSE_SEP} {negative} {settings.EOD_SEP} Answer Choices: {yesnooptionstring}"
                # text = re.sub(' +', ' ', text)
                # text = text + ' ' + post_prompt_selected #+ f' {response}'
                # output = 'no'
                # sequences.append({'input': text, 'outputs': output,
                #                     'index': idx, 'split': dp['split'],
                #                     'candidates': candidates, 'classes_in_options': candidates,
                #                     'dataset': dataset_reader.name})
                # idx += 1
            print('labelcount', labelcount)
            print(sequences[-2:])

        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
