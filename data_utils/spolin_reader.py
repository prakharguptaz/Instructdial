import os
from data_utils.data_reader import Dataset
import json


class SpolinDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.examples = []
        if split == 'train':
            filein = 'spolin-train.json'
        else:
            filein = 'spolin-valid.json'

        with open(os.path.join(data_path, filein)) as f:
            data = json.load(f)

        for typespolin, examples in data['yesands'].items():
            for dialogue in examples:
                dp = {'context': [dialogue['p']], 'response':dialogue['r'], 'type': typespolin}
                self.examples.append(dp)
