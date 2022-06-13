import json
import os
from data_utils.data_reader import Dataset


class ChitChatDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.idx = 0
        self.examples = []

        dir_path = os.path.join(data_path, split)
        filenames = next(os.walk(os.path.join(data_path, split)), (None, None, []))[2]

        for filename in filenames:
            with open(os.path.join(dir_path, filename)) as f:
                # dialogues.append(json.load(f))
                dialogues = json.load(f)
                for dialogue in dialogues:
                    self.examples.append(dialogue['turns'])
