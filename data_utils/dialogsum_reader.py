import json
from data_utils.data_reader import Dataset
from pathlib import Path

class DialogSumDataset(Dataset):
    def __init__(self, data_path: str, max_seq_length=512, split='test'):
        self.examples = []
         
        with open(f'{data_path}/dialogsum.{split}.jsonl') as f:
            for line in f.readlines():
                data = json.loads(line)
                if split == 'test':
                    self.examples.append({
                        'context': data['dialogue'],
                        'topic': data['topic1'],
                        'summary': data['summary1']
                    })
                else:
                    self.examples.append({
                        'context': data['dialogue'],
                        'topic': data['topic'],
                        'summary': data['summary']
                    })

        self.split = split
