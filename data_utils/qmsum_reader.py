import json
from data_utils.data_reader import Dataset
from pathlib import Path

class QMSumDataset(Dataset):
    def __init__(self, data_path: str, max_seq_length=512, split='test'):
        self.examples = []
        
        if split == 'dev':
            split = 'val'

        path = Path(f'{data_path}/ALL/{split}')
        for file in path.iterdir():
            with file.open() as f:
                data = json.load(f)        

                self.examples.append({
                    'topic_list': data['topic_list'],
                    'general_query_list': data['general_query_list'],
                    'specific_query_list': data['specific_query_list'],
                    'context': data['meeting_transcripts']
                })

        self.split = split