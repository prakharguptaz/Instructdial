import os
from data_utils.data_reader import Dataset
import csv

class GoEmotionsDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.examples = []

        with open(os.path.join(data_path, 'emotions.txt')) as f:
            self.emotion_classes = [l.strip() for l in f.readlines()]

        with open(os.path.join(data_path, '{}.tsv'.format(split))) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                emotions = [self.emotion_classes[int(i.strip())] for i in row[1].split(',')]

                # context is an empty string since there are no dialogues present in the dataset
                self.examples.append({
                    'context': [],
                    'response': row[0],
                    'emotions': emotions
                })
