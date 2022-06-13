import os
from data_utils.data_reader import Dataset


class SaferDialoguesDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.idx = 0
        self.examples = []

        with open(os.path.join(data_path, 'saferdialogues_dataset', split + '.txt')) as f:
            lines = [l.rstrip() for l in f.readlines()]

        print('len data', len(lines))
        for i, line in enumerate(lines):
            chattext = line.split('labels:')[0][5:].split('\\n')
            chattext = [x.strip() for x in chattext]
            recovery_response = line.split('labels')[1].split('\t')[0].strip()
            if recovery_response.startswith(':'):
                recovery_response = recovery_response[1:]
            self.examples.append({
                'context': chattext[:-1],
                'response': chattext[-1],
                'recovery_response': recovery_response
            })
