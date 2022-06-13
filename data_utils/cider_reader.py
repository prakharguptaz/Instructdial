import os
import json
import re
from data_utils.data_reader import Dataset


class CiderDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.examples = []
        self.split = 'train'

        with open(os.path.join(data_path, 'cider_main.json')) as f:
            data = json.load(f)
            for dialogue_data in data:
                utterances = [utt.strip() for utt in re.split(r'(A:|B:)', dialogue_data['utterances']) if
                              not re.match(r'(A:|B:)', utt.strip()) and len(utt.strip()) > 0]
                context = " ".join(utterances)
                context = utterances
                for relations in dialogue_data['triplets']:
                    contextrel = context + [f"The relation between {relations['head']} and {relations['tail']} is ?"]
                    answer = relations['relation']
                    if type(answer) is str:
                        answer_tokens = re.findall('[A-Z][^A-Z]*', answer)
                        answer = ' '.join([x.lower() for x in answer_tokens])
                    self.examples.append({"utterances": utterances, "data": dialogue_data['triplets'], 'context': contextrel, 'answer': answer})
