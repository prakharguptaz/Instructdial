import os
import json
from data_utils.data_reader import Dataset


class MultiwozDstDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.idx = 0
        self.examples = []

        if split == 'train':
            self.examples = self.process_data('/multiwoz-fine-processed-train.json')
        elif split == 'test':
            self.examples = self.process_data('/multiwoz-fine-processed-test.json')
        else:
            self.examples = self.process_data('/multiwoz-fine-processed-dev.json')

    def process_data(self, filename):
        with open(self.data_path + filename) as f:
            data = json.load(f)

        examples = []
        for session in data:
            context = []
            for turn in session:
                usr_turn = turn['user'].strip('<sos_u>').strip('<eos_u>').strip()
                resp_turn = turn['resp'].strip('<sos_r>').strip('<eos_r>').strip()
                context.append(usr_turn)

                examples.append({
                    "context": context[:],
                    "dial_id": turn['dial_id'],
                    "user": usr_turn,
                    "response": resp_turn,
                    "state": turn['bspn'].strip('<sos_b>').strip('<eos_b>'),
                    "turn_domain": turn['turn_domain'],
                    "turn_num": turn['turn_num'],
                    "db": turn['db']})

                context.append(resp_turn)

        return examples
