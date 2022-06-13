import os
from data_utils.data_reader import Dataset
import json
import copy

class DialogREDataset(Dataset):
    def __init__(self, data_path, split):
        self.idx = 0
        self.examples = []
        with open(f'{data_path}/{split}.json') as f:
            data = json.load(f)
            class_set = set()
            for idx, dp in enumerate(data):
                context = dp[0]
                relations = dp[1]
                formatted_relations = []
                for rel in relations:
                    newrel = copy.deepcopy(rel)
                    newrel['r'] = []
                    rels = rel['r']
                    for r in rels:
                        r = r.replace('per:', 'person:').replace('gpe:', 'location:').replace('org:', 'organization:')
                        newrel['r'].append(r)
                        class_set.add(r)
                    formatted_relations.append(newrel)
                self.examples.append({"context":context, "index":idx, "relations":formatted_relations})
            self.intent_classes = list(class_set)
            # print(self.intent_classes)



