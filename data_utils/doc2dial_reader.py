import os
from data_utils.data_reader import Dataset
import json
import re


class Doc2DialDataset(Dataset):
    def __init__(self, data_path, split='test'):
        self.idx = 0
        self.examples = []

        with open(os.path.join(data_path, 'doc2dial_doc.json'.format(split))) as f:
            documents = json.load(f)

        if split != 'train':
            split = 'validation'

        with open(os.path.join(data_path, 'doc2dial_dial_{}.json'.format(split))) as f:
            data = json.load(f)

        for topic, articles in data['dial_data'].items():
            for article, dialogues in articles.items():
                for dialogue in dialogues:
                    context = []
                    for turn in dialogue['turns']:
                        if turn['role'] == 'agent':
                            document_spans = list(documents['doc_data'][topic][article]['spans'].values())
                            sp_ids = [e['sp_id'] for e in turn['references']]

                            docs = []
                            for span in document_spans:
                                if span['id_sp'] in sp_ids:
                                    docs.append(span['text_sp'])
                            self.examples.append({
                                'context': context[:],
                                'response': turn['utterance'],
                                'doc': docs
                            })
                        context.append(turn['utterance'])
