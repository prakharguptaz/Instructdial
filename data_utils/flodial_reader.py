import json
import os
from data_utils.data_reader import Dataset
import re


def read_schema(data_path, filename):
    with open(os.path.join(data_path, 'FloDial-dataset', 'knowledge-sources', filename)) as f:
        data = json.load(f)
    return data


class FlodialDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.examples = []

        with open(os.path.join(data_path, 'FloDial-dataset', 'dialogs', 'dialogs.json')) as f:
            data = json.load(f)

        with open(os.path.join(data_path, 'FloDial-dataset', 'dialogs', 'u-flo.json')) as f:
            splits = json.load(f)

        if split == 'train':
            idxs = set(splits['trn'])
        else:
            idxs = set(splits['val'])

        schemas = {
            'brake_problem': read_schema(data_path, 'brake_problem.json'),
            'car_electrical_failure': read_schema(data_path, 'car_electrical_failure.json'),
            'car_wont_start': read_schema(data_path, 'car_wont_start.json'),
            'engine_overheats': read_schema(data_path, 'engine_overheats.json'),
            'laptop_drive': read_schema(data_path, 'laptop_drive.json'),
            'laptop_overheating': read_schema(data_path, 'laptop_overheating.json'),
            'lcd_problem': read_schema(data_path, 'lcd_problem.json'),
            'power': read_schema(data_path, 'power.json'),
            'ticking': read_schema(data_path, 'ticking.json'),
            'wireless': read_schema(data_path, 'wireless.json')
        }

        for id, dialog in data.items():
            if id not in idxs:
                continue
            schema_data = schemas[dialog['flowchart']]
            context = []
            for utterence in dialog['utterences']:
                text = utterence['utterance']
                if utterence['speaker'] == 'agent':
                    schema = {}
                    if 'grounded_doc_id' in utterence:
                        grounded_doc_id = utterence['grounded_doc_id']

                        m1 = re.match(r'chart-([0-9])', grounded_doc_id)
                        m2 = re.match(r'faq-([0-9])', grounded_doc_id)

                        if m1:
                            idx = m1.group(1)
                            schema = schema_data['nodes'][idx]
                        if m2:
                            idx = m2.group(1)
                            schema = schema_data['supporting_faqs'][int(idx)]

                    self.examples.append({
                        'context': context[:],
                        'response': text,
                        'schema': json.dumps(schema)
                    })
                context.append(text)
