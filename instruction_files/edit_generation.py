'''
Multiple operations borrowed from https://github.com/iitmnlp/EvalEval/
'''

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

import random
import nltk
from nltk.tokenize import sent_tokenize
from checklist.perturb import Perturb
import re
import spacy
from checklist.editor import Editor
from checklist.perturb import Perturb


instruction_dict = {
	"id": "edit",
    "Definitions": ["In this task you will be shown a conversation context and a response. You need to edit the provided response so that it becomes coherent to the conversation based on the context.",
                    "Read the dialogue and the provided response to convert it into a coherent and fluent response.",
                    "Rephrase the provided response so that it is meaningful and relevant to the dialogue.",
                    "Edit the provided response into a response that is fluent and coherent to the dialogue context."],
    "Positive Examples": [
       
    ]
}



class TransformResponses:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = self.nlp.Defaults.stop_words
        
    def remove_punct(self, sent):
        return re.sub(r'[^\w\s]', ' ', sent) 

    # def typos(self, sent):
    #     return Perturb.add_typos(sent)

    # def contractions(self, sent):
    #     x = Perturb.contract(sent)
    #     return x if x != sent else sent

    # def expansions(self, sent):
    #     x = Perturb.expand_contractions(sent)
    #     return x if x != sent else sent

    # def add_negation(self, sent):
    #     try:
    #         x = Perturb.add_negation(self.nlp(sent)) 
    #         return x if x != None else sent
    #     except:
    #         return sent


    def jumble(self, sent):
        tokens = [i.text for i in self.nlp(sent)]
        random.shuffle(tokens)
        return ' '.join(tokens)
    
    def swap_tokens(self, sent):
        tokens = [i.text for i in self.nlp(sent)]
        # random.shuffle(tokens)
        random_index = random.choice(range(len(tokens)))
        random_index2 = random.choice(range(len(tokens)))
        tokens[random_index], tokens[random_index2] = tokens[random_index2], tokens[random_index]
        return ' '.join(tokens)

    def drop_stopwords(self, sent):
        x  = [word.text for word in self.nlp(sent) if not word.text in self.stopwords]
        return ' '.join(x)
    
    # def synonym_adjective(self, sent):
    #     pos = nltk.pos_tag(nltk.word_tokenize(sent))
    #     flag = 0
    #     sen =[]
    #     for i in range(len(pos)):
    #         w, p = pos[i]
    #         if p in ['JJ', 'JJR', 'JJS']:
    #             try:
    #                 syn = Editor().synonyms(sent, w)
    #             except:
    #                 syn = []
    #             if len(syn) > 0:
    #                 sen.append(syn[0])
    #                 flag = 1
    #             else:
    #                 sen.append(w)
    #         else:
    #             sen.append(w)
    #     if flag == 1:
    #         out = " ".join(x for x in sen)
    #         return out
    #     return sent

    # def antonym_adjective(self, sent):
    #     pos = nltk.pos_tag(nltk.word_tokenize(sent))
    #     flag = 0
    #     sen =[]
    #     for i in range(len(pos)):
    #         w, p = pos[i]
    #         if p in ['JJ', 'JJR', 'JJS']:
    #             try:
    #                 syn = Editor().antonyms(sent, w)
    #             except:
    #                 syn = []
    #             if len(syn) > 0:
    #                 sen.append(syn[0])
    #                 flag = 1
    #             else:
    #                 sen.append(w)
    #         else:
    #             sen.append(w)
    #     if flag == 1:
    #         out = " ".join(x for x in sen)
    #         return out
    #     return sent

#     def hyponyms(self, sent):
#         pos = nltk.pos_tag(nltk.word_tokenize(sent))
#         sen = []
#         flag = 0
#         for i in range(len(pos)):
#             w, p = pos[i]
#             if p in ['NN','NNP','VB','VBP']:
#                 try:
#                     syn = Editor().hyponyms(templates =sent, word = w)
#                 except:
#                     syn = []
#                 if len(syn) > 0:
#                     sen.append(syn[0])
#                     flag += 1
#                 else:
#                     sen.append(w)
#             else:
#                 sen.append(w)
#         if flag > 0:
#             out = " ".join(x for x in sen)
#             return out
#         return sent

#     def subject_verb_dis(self, sent):
#         cases = {'was':'were', 
#                 'were':'was', 
#                 'is':'are',
#                 'are':'is', 
#                 'has':'have',
#                 'have':'has',
#                 'does':'do',
#                 'do':'does'}
#         sentence =''
#         doc = self.nlp(sent)
#         for i in doc:
#             if i.pos_ =="AUX":
#                 try:
#                     w = cases[i.text]
#                 except:
#                     w =i.text
#                 sentence  = sentence + w + ' '
#             else:
#                 sentence = sentence + i.text + ' '
#         return sentence.strip()

    
    def repeat_phrases(self, sent):
        pos = nltk.pos_tag(nltk.word_tokenize(sent))
        sen = []
        l = len(pos)
        rep_word = ''
        flag = 0
        for i in range(l-1):
            w, p = pos[i]
            if i< l*0.25:
                rep_word += " " + w
                flag = 1
                sen.append(w)
            else:
                sen.append(w)
        sen.append(pos[l-1][0])
        sen.append(rep_word)
        if flag==1: 
            out = " ".join(w for w in sen)
            return out
        return sent



    def change_names(self, sent):
        text = self.nlp(sent)
        x = Perturb.perturb([text], Perturb.change_names, n=1).data
        return sent if  x==[] else x[0][1]


    def drop_phrases(self, sent):
        pos = nltk.pos_tag(nltk.word_tokenize(sent))
        sen = []
        l = len(pos)
        flag = 0
        le = round(l*0.6)
        if len(sent)<5:
            x = 0
        else:
            x = random.randint(0,le-1)
        y = 0
        for i in range(l-1):
            w, p = pos[i]
            if x<=i and y < round(l*0.2): 
                y+=1
                flag = 1
                continue
            else:
                sen.append(w)
        sen.append(pos[l-1][0])
        if flag==1: 
            out = " ".join(w for w in sen)
            return out 
        return sent

    def insert_phrases(self, sent, sent2):
        sent2_tokens = sent2.split()
        if len(sent2_tokens)<3:
            init = 0
            end = 1
        else:
            init = random.choice(range(int(len(sent2_tokens)/2)))
            end = random.choice(range(init, len(sent2_tokens)))
        sent2_tokens = sent2_tokens[init:end]
        if len(sent2_tokens)>8:
            sent2_tokens = sent2_tokens[:8]
        sent2_phrase = ' '.join(sent2_tokens)
        pos = nltk.pos_tag(nltk.word_tokenize(sent))
        sen = []
        l = len(pos)
        flag = 0
        le = round(l*0.6)
        x = random.randint(0,le-1)
        y = 0
        for i in range(l-1):
            w, p = pos[i]
            if x<=i and y < round(l*0.2) and flag != 1: 
                y+=1
                flag = 1
                sen.append(sent2_phrase)
                continue
            else:
                sen.append(w)
        sen.append(pos[l-1][0])
        if flag==1: 
            out = " ".join(w for w in sen)
            return out 
        return sent    
    
    def repeat_sentences(self, sent):
        toks = sent_tokenize(sent)
        sent = []
        l = len(toks)
        i = 0
        while i<l:
            sent.append(toks[i])
            i+=1
        sent.append(toks[0])
        out = " ".join(x for x in sent)
        return out if out !=sent  else sent
    
    def sentence_reorder(self, sent):
        text_split = [i.text for i in self.nlp(sent).sents]
        random.shuffle(text_split)
        return " ".join(text_split)


def get_random_transform(poison, sent, sent1):
    # object_methods = [method_name for method_name in dir(poison)
    #                   if callable(getattr(poison, method_name))]
    object_methods = dir(poison)
    found_one = False
    num_trial = 10
    result = sent
    chosen_method = 'no operation'
    while not found_one or num_trial<0:
        num_trial-=1
        if num_trial<0:
            break
        # random.shuffle(object_methods)
        chosen_method= random.choice(object_methods)
        if chosen_method in ['stopwords', 'nlp'] or '__' in chosen_method:
            continue
        # print(chosen_method)
        found_fn = getattr(poison, chosen_method)
        if chosen_method=='insert_phrases':
            result = found_fn(sent, sent1)
        else:
            result = found_fn(sent)
        if result!=sent:
            found_one = True
            
    return chosen_method, result
        

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
            print(dataset_reader)
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
            poison = TransformResponses()

            for dp in tqdm(datapoints):
                index = dp.get('index', -1)
                split = dp.get('split', 'unspecified')

                if 'sum' in dataset_reader.name.lower():
                    if '\n#' in dp['context']:
                        dp['response'] = dp['context'].split('\n')[-1]
                        dp['context'] = dp['context'].split('\n')[:-1]

                if type(dp['context']) is str:
                    dp['context'] = [dp['context']]
                    # print('context str format found')
                    # import pdb;pdb.set_trace()
                context = (' '+settings.EOT_SEP+ ' ').join(dp['context'][-settings.MAX_CONTEXT_NUMUTTERANCE:])
                if 'response' not in dp:
                    print(dp)
                    import pdb;pdb.set_trace()
                response = dp['response']
                
                sent1 = random.choice(datapoints).get('response', None)
                if sent1 == None:
                    sent1 = random.choice(datapoints).get('context', 'no good choice now')[0]

                chosen_transform, condition_response_str = get_random_transform(poison, response, sent1)

                if ''.join(condition_response_str.split())==''.join(response.split()):
                    continue

                context_str = ' '.join(context.split()[-settings.MAX_DIALOGUE_LENGTH:])
                if len(context_str)==0:
                    continue
                    # dp.pop('knowledge', None)
                    # print(dp)
                text =  settings.CONTEXT_SEP +" "+ context_str + " " + settings.RESPONSE_SEP + " " + condition_response_str + " " + settings.EOD_SEP 
                post_prompts = [settings.QUESTION_SEP+" Given this context and response provided, the edited response is",
                                settings.QUESTION_SEP+" Generate the rephrased response with the provided context",
                                settings.QUESTION_SEP+" Given this context generate the edited version of the response that is coherent",
                                settings.QUESTION_SEP+" Here is a response which is a coherent rephrase of the given response"]
                
                text = text +' '+ random.choice(post_prompts)
                text = re.sub(' +', ' ', text)
                output = response
                sequences.append({'text':text, 'output': output, 'metadata':{'context':dp['context'], 'condition_response_str':condition_response_str, 'chosen_transform':chosen_transform}, 'index':index, 'split':split, 'dataset':dataset_reader.name})

        new_defs = []
        for definition in instruction_dict['Definitions']:
            extra_actions = ['improve', 'revise', 'polish', 'modify', 'adapt', 'rewrite', 'reword', 'redraft', 'rephrase', 'alter']
            for i in range(10):
                chosen_action = random.choice(extra_actions)
                rephrase = definition.replace('edit',chosen_action)
                rephrase = definition.replace('convert',chosen_action)
                rephrase = definition.replace('Rephrase',chosen_action)
                rephrase = definition.replace('Edit',chosen_action)

                # print(rephrase)
                if rephrase !=definition and rephrase not in new_defs:
                    new_defs.append(rephrase.capitalize())
        instruction_dict['Definitions']+=new_defs
        return (sequences, instruction_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
