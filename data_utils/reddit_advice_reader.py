import os
from data_utils.data_reader import Dataset
import json
import re
import random


class RedditAdviceDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.examples = []

        with open(os.path.join(data_path, 'redditadvice2019.jsonl')) as f:
            json_list = list(f)

        for json_str in json_list:
            line = json.loads(json_str)

            if line['split'] == split:

                comments = []
                for c in line['good_comments']:
                    comment = c['body']
                    if '\n' in comment:
                        comment = re.sub('\n\n+', '\n', comment)
                        comment = re.sub('  +', ' ', comment)
                        comment = re.split('\n', comment)[0]
                    comments.append(comment)

                self.examples.append({
                    'context': [line['selftext']],
                    'topic': line['subreddit'],
                    'comments': comments,
                    'response': random.choice(comments)})
