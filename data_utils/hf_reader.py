import csv
import logging
import json
import numpy as np
import os
import pickle
import re
import copy
from collections import defaultdict, Counter
# from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict
import random
from constants import SPECIAL_TOKENS
#Hugging Face datasets
from datasets import list_datasets, load_dataset, list_metrics, load_metric


BADSENT = r'\.(\w)'
ALL_COUNTS = Counter()

ACTS = [
    'no_act',
    'inform',
    'question',
    'directive',
    'commissive',
]

EMOTIONS = [
    'no_emotion',
    'anger',
    'disgust',
    'fear',
    'happiness',
    'sadness',
    'surprise',
]

TOPICS = [
    'no_topic',
    'ordinary_life',
    'school_life',
    'culture_and_educastion',
    'attitude_and_emotion',
    'relationship',
    'tourism',
    'health',
    'work',
    'politics',
    'finance',
]


def cleanup_text(text):
    text = text.strip()

    # Prefer non-unicode special character
    SWITCH_LIST = [
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u201d", '"'),
        ("\u201c", '"'),
        ("\u2014", "--"),
        ("\u2013", "--"),
        ("\u3002", ". "),
        ("\u2032", "'"),
        ("\u3001", ", "),
    ]
    for before, after in SWITCH_LIST:
        text = text.replace(before, after)

    # fix some broken sentence tokenization
    text = re.sub(BADSENT, r' . \1', text)

    ALL_COUNTS.update([t for t in text.split() if len(t) == 1 and ord(t) > 127])
    return text.strip()


class Dataset():
    def __init_(self):
        self.idx=0
        self.examples = []
    
    def get_next(self):
        if self.idx>=len(self.examples):
            return None
        dp = self.examples[self.idx]
        self.idx+=1

        return dp

class DailyDialogDataset(Dataset):
    def __init__(self, split='train'):
        self.examples = []
        self.dataset = load_dataset('roskoN/dailydialog')
        self.datasetname = 'dailydialog'
        self.act_classes = ACTS
        data = self.dataset[split]
        self.emotion_classes = set()
        for idx, dp in enumerate(data):
            context_data = []
            utterances = [cleanup_text(x) for x in dp['utterances']]
            emotions = [EMOTIONS[int(x)] for x in dp['emotions']]
            acts = [ACTS[int(x)] for x in dp['acts']]
            for i in range(0,len(utterances)):
                context = utterances[:i]
                response = utterances[i]
                emotion = emotions[i]
                act = acts[i]
                newdp = {'context': context, 'response':response , 'emotions':emotion, 'dialogue_act':act, 'act_classes':[act], 'utterance_id':i, 'context_id': idx}
                self.emotion_classes.add(emotion)
                context_data.append(newdp)
            self.examples+=context_data

        self.emotion_classes = list(self.emotion_classes)
        self.idx=0



class EmpatheticDialoguesDataset(Dataset):
    def __init__(self, split='train'):
        self.examples = []
        self.dataset = load_dataset('empathetic_dialogues')
        self.datasetname = 'empathetic_dialogues'
        data = self.dataset[split]
        all_datadict= defaultdict(list)
        for idx, dp in enumerate(data):
            # context_data = []
            all_datadict[dp['conv_id']].append(dp)

        for k in all_datadict.keys():
            utterances = all_datadict[k]
            history = []
            for i in range(len(utterances)):
                dp = utterances[i]
                context_id = dp['conv_id']
                response = dp['utterance'].replace('_comma_', ',')
                emotion = dp['context']
                conv_prompt = dp['prompt']
                context = ' '.join(history[:])
                context = re.sub('  +', ' ', context.strip())
                if len(context) > 0:
                    if i%2==0:
                        newdp = {'context': history[:],
                             'response':response ,
                             'emotions': [emotion],
                             'utterance_id':i,
                             'context_id': context_id,
                             'conv_prompt':conv_prompt}
                        self.examples.append(newdp)
                history.append(response)


        self.idx=0



class Convai2Dataset(Dataset):
    def __init__(self, split='train'):
        self.examples = []
        self.dataset = load_dataset('conv_ai_2')
        self.datasetname = 'convai2'
        data = self.dataset['train']


        all_datadict= defaultdict(list)
        for idx, dp in enumerate(data):
            # context_data = []
            # all_datadict[dp['dialog_id']].append(dp)
            if 3<len(dp['bot_profile'])<6 and len(dp['bot_profile'][0][0])==1:
                dp['bot_profile'] = [''.join(profile_sent) for profile_sent in dp['bot_profile']]                                  
                dp['user_profile'] = [''.join(profile_sent) for profile_sent in dp['user_profile']]
            utterances = dp['dialog']
            history = []
            for i in range(len(utterances)):
                utt = utterances[i]
                context_id = dp['dialog_id']
                response = utt['text']
                sender = utt['sender']
                sender_class = utt['sender_class']
                if sender_class == 'Bot':
                    profile = dp['bot_profile']
                if sender_class == 'Human':
                    profile = dp['user_profile']
                if len(response)>10 and 'This is your profile' not in response:
                    newdp = {'context': history[:], 'personality':profile, 'user_profile':dp['user_profile'], 'bot_profile':dp['bot_profile'], 'response':response, 'sender':sender, 'sender_class':sender_class, 'utterance_id':i, 'context_id': context_id}
                    self.examples.append(newdp)                 
                history.append(response)

        print('len data', len(self.examples))

        num_val = len(self.examples) // 10
        if split == 'train':
            self.examples = self.examples[:-num_val*2]
        elif split == 'dev':
            self.examples = self.examples[-num_val*2:-num_val]
        elif split == 'test':
            self.examples = self.examples[-num_val:]
        print('Using', len(self.examples))


        self.idx=0


class PersonachatDataset(Dataset):
    def __init__(self, split='train', return_candidates=False):
        self.examples = []
        if split == 'test':
            split = 'validation'
        self.dataset = load_dataset('bavard/personachat_truecased')
        self.datasetname = 'personachat'
        data = self.dataset[split]
        all_datadict= defaultdict(list)
        for idx, dp in enumerate(data):
            history = dp['history']
            context_id = dp['conv_id']
            uid = dp['utterance_idx']
            candidates = dp['candidates']
            personality = dp['personality']
            response = candidates[-1]
            if len(response)>10:
                newdp = {'context': history[:], 'personality':personality, 'response':response, 'utterance_id':uid, 'context_id': context_id}
                if return_candidates:
                    newdp['candidates'] = candidates
                self.examples.append(newdp)                    
            history.append(response)

        print('len data', len(self.examples))

        self.idx=0



class SamsumDataset(Dataset):
    def __init__(self, split='train', return_candidates=False):
        self.examples = []
        self.dataset = load_dataset('samsum')
        self.datasetname = 'samsum'
        data = self.dataset[split]
        all_datadict= defaultdict(list)
        for idx, dp in enumerate(data):
            history = dp['dialogue'].split('\r\n')
            context_id = dp['id']
            summary = dp['summary']
            newdp = {'context': history[:], 'summary':summary, 'context_id': context_id}
            self.examples.append(newdp)                    

        print('len data', len(self.examples))

        self.idx=0

class TimeDialDataset(Dataset):
    def __init__(self, split='test', return_candidates=False):
        self.examples = []
        self.dataset = load_dataset('time_dial')
        self.datasetname = 'time_dial'
        data = self.dataset[split]
        all_datadict= defaultdict(list)
        for idx, dp in enumerate(data):
            newdp = copy.deepcopy(dp)
            history = dp['conversation']
            # context_id = dp['id']
            # summary = dp['summary']
            newdp['context'] = history[:]
            newdp['answer'] = newdp['correct1']
            newdp['answer2'] = newdp['correct2']
            newdp['options'] = [newdp['incorrect1'], newdp['incorrect2'], newdp['correct1']]
            random.shuffle(newdp['options'])
            del newdp['conversation']
            self.examples.append(newdp)

        print('len data', len(self.examples))
        print(self.examples[0])

        self.idx=0
