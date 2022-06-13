import json
from pathlib import Path
import numpy as np
from data_utils.data_reader import Dataset
import math
import random
import settings

class EvalDataset(Dataset):
    def __init__(self, base_dir: Path, split='dev', max_perdataset = 50000):
        self.examples = []
        self.max_perdataset = max_perdataset
        self.load_dstc10_format(base_dir)
        
        self.split = split
        print('Total',len(self.examples))
        random.shuffle(self.examples)

        num_val = len(self.examples) // 10
        # if split == 'train':
        #     self.examples = self.examples[:-num_val*4]
        # elif split == 'dev':
        #     self.examples = self.examples[-num_val*4:-num_val*2]
        # elif split == 'test':
        #     self.examples = self.examples[-num_val*2:]
        if split == 'train':
            self.examples = self.examples[:-num_val*2]
        elif split == 'dev':
            self.examples = self.examples[-num_val*2:-num_val*1]
        elif split == 'test':
            self.examples = self.examples[-num_val*1:]
        else:
            self.examples = self.examples
        print('Using', len(self.examples))
   
    def parse_data(self, data, dataset_name):
        all_scores = []

        samples = []
        for sample in data:
            if 'context' in sample:
                # fact, reference
                context, response = sample['context'], sample['response']
                context = context.strip().split('\n')
                response = response.strip()
            else:
                # fed_dialog
                dialog = sample['dialog']
                context = []
                for turn in dialog[:-1]:
                    context.append(turn['text'])
                response = dialog[-1]['text']
            persona = None
            knowledge = None
            if 'persona' in dataset_name or 'convai' in dataset_name or 'topical-usr' in dataset_name:
                if 'fact' in sample and dataset_name == 'persona-usr':
                    persona = sample['fact'].replace('your persona: ', '').split('\n')
                if 'fact' in sample and dataset_name == 'topical-usr':
                    knowledge = sample['fact'][-settings.MAX_DOCUMENT_LENGTH:]
                if dataset_name == 'persona-see':
                    if sample['dialog'][-1]['speaker'] == 'model':
                        persona = sample['model_persona'].split('\n')
                    else:
                        persona = sample['human_persona'].split('\n')

            if 'model' in sample:
                model = sample['model']
            else:
                model = 'Dummy'

            annotation = sample['annotations']
            score = {}
            qualities = []
            for quality, score_list in annotation.items():
                score[quality.lower()] = np.mean(score_list)
                qualities.append(quality.lower())

                all_scores.append(np.mean(score_list))
            
            # check if response, context are empty
            if ' '.join(context).strip() == '':
                context = ['']
                print('No ctx')
            
            if response.strip() == '':
                response = ''
                print('No response')
            
            samples.append(
                {
                    'd_id': sample['dialogue_id'],
                    'context': context,
                    'response': response,
                    'reference': '',
                    'score': score,
                    'persona': persona,
                    'knowledge': knowledge,
                    'model': model,
                    'qualities': qualities,
                    'dataset_name':dataset_name
                }
            )

        min_score = int(min(all_scores))
        max_score = math.ceil(max(all_scores))
        for sample in samples:
            sample['score_min'] = min_score
            sample['score_max'] = max_score
        return samples

    def load_dstc10_format(self, base_dir):
        datasets = [
            'fed-dial',
            'dstc6',
            'fed-turn',
            'persona-usr',
            'empathetic-grade',
            'dstc7',
            'convai2-grade',
            'dailydialog-zhao',
            'dailydialog-grade',
            'persona-see',
            'dailydialog-gupta',
            'persona-zhao',
            'humod',
            'topical-usr'
        ]
        base_dir = Path(base_dir + '/dstc10_format')
        for dataset in datasets:
            with (base_dir / f'{dataset}_eval.json').open() as f:

                data = json.load(f)
                samples = self.parse_data(data, dataset)
                samples = samples[:self.max_perdataset]
                self.examples.extend(samples)
                print(f'{dataset} Num {len(samples)}')

if __name__ == '__main__':
    data = load_dstc10_data('.')
    print(data['data_nums'])
    with open('dstc10_data.json', 'w') as f:
        json.dump(data, f)
