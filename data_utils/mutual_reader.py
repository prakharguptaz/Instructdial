import json
from data_utils.data_reader import Dataset
from pathlib import Path
import re

option_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

class MutualDataset(Dataset):
    def __init__(self, data_path: str, max_seq_length=512, split='test'):
        self.examples = []
        self.split = split

        if split == 'train' or split == 'dev':
            path = Path(f'{data_path}/train')
        else: # test directory have no label so we use dev as test
            path = Path(f'{data_path}/dev') 
        for data in path.iterdir():
            with open(data) as f:
                data = json.loads(f.readline())

                data['context'] = [utt.strip() for utt in re.split(r'(f :|m :)', data['article']) if not re.match(r'(f :|m :)', utt.strip()) and len(utt.strip()) > 0]
                data['options'] = [do.split('m : ')[-1].split('f : ')[-1] for do in data['options']]
                data['answer'] = data['options'][option_map[data['answers']]]
                self.examples.append(data)

        if split == 'train':
            num_examples = len(self.examples)
            self.examples = self.examples[:-num_examples//10]
        elif split == 'dev':
            num_examples = len(self.examples)
            self.examples = self.examples[-num_examples//10:]
