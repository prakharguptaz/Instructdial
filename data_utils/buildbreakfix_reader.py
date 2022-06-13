import os
from data_utils.data_reader import Dataset
import json


class BuildBreakFixDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.idx = 0
        self.examples = []

        with open(os.path.join(data_path, 'dialogue_safety', 'multi_turn_safety.json')) as f:
            data = json.load(f)

        data = data[split]
        for d in data:
            context = [l.strip() for l in d['text'].split("\n")]
            label = d['labels'][0]

            toxic = False
            if label == '__notok__':
                toxic = True

            self.examples.append({
                'context': context[:-1],
                'response': context[-1],
                'toxic': toxic
            })