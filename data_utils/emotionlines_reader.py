import os
from data_utils.data_reader import Dataset
import csv
from collections import defaultdict
import re


class EmotionLinesDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.emotion_classes = set([])
        self.examples = []

        with open(os.path.join(data_path, '{}_sent_emo.csv'.format(split))) as f:
            rows = csv.DictReader(f)

            context = []
            dialogue_id = '0'

            for row in rows:
                if row['Dialogue_ID'] != dialogue_id:
                    dialogue_id = row['Dialogue_ID']
                    context = []

                emotions = [row['Emotion']]
                self.emotion_classes.update(emotions)
                context_str = ' '.join(context)
                context_str = re.sub('  +', ' ', context_str.strip())
                if len(context_str) > 0:
                    self.examples.append({'context': context[:],
                                          'emotions': emotions,
                                          'response': row['Utterance']})
                context.append(row['Utterance'])
        self.emotion_classes = list(self.emotion_classes)
