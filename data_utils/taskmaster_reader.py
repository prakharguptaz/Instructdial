import os
from data_utils.data_reader import Dataset
import json


class TaskMasterDataset(Dataset):
    def __init__(self, data_path):
        self.idx = 0
        self.examples = []

        categories = ['flights', 'food-ordering', 'hotels', 'movies', 'music', 'restaurant-search', 'sports']
        for category in categories:
            with open(os.path.join(data_path, category + '.json')) as f:
                data = json.load(f)
                for conv in data:
                    if len(conv['slots']) > 0:
                        print()
                    conv['category'] = category
                    self.examples.append(conv)
