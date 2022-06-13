import json
from data_utils.data_reader import Dataset
from pathlib import Path

class TimeDialDataset(Dataset):
    def __init__(self, data_path: str, max_seq_length=512):
        self.examples = []
        self.split = 'train'

        with open(f'{data_path}/test.json') as f:
            data = json.load(f)
            for sample in data:
                self.examples.append({
                    'dialog': sample['conversation'],
                    'answer': sample['correct1'],
                    'correct': [sample['correct1'], sample['correct2']],
                    'incorrect': [sample['incorrect1'], sample['incorrect2']]

                })

