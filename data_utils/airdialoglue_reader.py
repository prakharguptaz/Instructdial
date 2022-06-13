import os
from data_utils.data_reader import Dataset
import json


class AirDialogueDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.examples = []

        with open(os.path.join(data_path, split + '_data.json')) as f:
            self.examples = [json.loads(l) for l in f.readlines()]
