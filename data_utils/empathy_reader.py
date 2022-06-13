import os
from data_utils.data_reader import Dataset
import csv
import random


class EmpathyDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0

        if split=='test':
            data_path = './datasets/empathy/sample_test_input.csv'
        posts = {}
        with open(os.path.join(data_path)) as f:
            rows = csv.DictReader(f)
            for row in rows:
                sp_id = row.get('sp_id', -1)

                if 'sp_id' in row and row['sp_id'] not in posts:
                    posts[sp_id] = {'text': row['seeker_post'], 'context': [row['seeker_post']], 'comments': []}
                if sp_id==-1:
                    posts[sp_id] = {'text': row['seeker_post'], 'context': [row['seeker_post']], 'comments': []}
                    

                posts[sp_id]['comments'].append(row['response_post'])
                posts[sp_id]['response'] = random.choice(posts[sp_id]['comments'])

        self.examples = list(posts.values())


"""
                    posts[sp_id] = {'context': row['seeker_post'],
                                    'comments': []}
                posts[sp_id]['comments'].append(row['response_post'])
"""
