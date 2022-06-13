import os
from data_utils.data_reader import Dataset
import csv


class BadDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.idx = 0
        self.examples = []

        with open(os.path.join(data_path, 'bot_adversarial_dialogue', 'dialogue_datasets',
                               'bot_adversarial_dialogue_datasets_with_persona', split + '.txt')) as f:
            lines = [l.rstrip() for l in f.readlines()]

        for l in lines:
            context = l.split('labels:')[0][5:].split('\\n')
            context = [x.strip() for x in context]

            label = l.split('labels:')[1].split()[0]
            toxic = False
            if label == '__notok__':
                toxic = True

            self.examples.append({
                'context': context[:-1],
                'response': context[-1],
                'toxic': toxic
            })
