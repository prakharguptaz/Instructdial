import os
from data_utils.data_reader import Dataset
import re


class DealDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.examples = []

        with open(os.path.join(data_path, split + '.txt')) as f:
            for line in f.readlines():
                m = re.match(r'.*<dialogue>(.+)</dialogue>.*', line)
                text = m.group(1)
                text = re.sub('<selection>|THEM:|YOU:', ' ', text)
                text = re.sub('  +', ' ', text).strip()
                turns = [t.strip() for t in text.split('<eos>') if len(t.strip()) > 0]

                agreement = True
                if '<no_agreement>' in line:
                    agreement = False
                self.examples.append({
                    'context': turns[:],
                    'agreement': agreement
                })

        # self.examples = self.examples[1::2]
