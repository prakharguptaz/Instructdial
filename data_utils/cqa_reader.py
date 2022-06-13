from typing import List
import json
from data_utils.data_reader import Dataset

class CQADataset(Dataset):
    def __init__(self, data_path: str, data_type:str, max_seq_length=512, split='train'):
        if data_type == 'coqa':
            if split == 'train' or split == 'dev':
                data_path = f'{data_path}/coqa-train-v1.0.json'
            elif split == 'test':
                data_path = f'{data_path}/coqa-dev-v1.0.json'

            with open(data_path) as f:
                data = json.load(f)['data']
            self.process_coqa(data)

            if split == 'train':
                num_data = len(self.examples)
                self.examples = self.examples[:-num_data//10]
            elif split == 'dev':
                num_data = len(self.examples)
                self.examples = self.examples[-num_data//10:]
                print(num_data, len(self.examples))
        elif data_type == 'quac':
            if split == 'train' or split == 'dev':
                data_path = f'{data_path}/train_v0.2.json'
            elif split == 'test':
                data_path = f'{data_path}/val_v0.2.json'
                
            with open(data_path) as f:
                data = json.load(f)['data']
            self.process_quac(data)

            if split == 'train':
                num_data = len(self.examples)
                self.examples = self.examples[:-num_data//10]
            elif split == 'dev':
                num_data = len(self.examples)
                self.examples = self.examples[-num_data//10:]

        self.split = split
    
    def process_coqa(self, data: List[dict]):
        self.examples = []
        for article in data:
            doc = article['story']
            context = []
            
            for q, ans in zip(article['questions'], article['answers']):
                question = q['input_text']
                span_answer = ans['span_text']
                answer = ans['input_text']
                
                context += [question]#+= f'{question} '
            
                self.examples.append({
                    'document': doc,
                    'question': question,
                    'span_answer': span_answer,
                    'answer': answer,
                    'context': context[:]
                })
                
                # context += f'{answer}. '
                context += [answer]

    def process_quac(self, data: List[dict]):
        self.examples = []
        for article in data:
            for paragraph in article['paragraphs']:
                doc = paragraph['context']
                context = []

                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = qa['orig_answer']['text']

                    context += [question]#+= f'{question} '
                
                    self.examples.append({
                        'document': doc,
                        'question': question,
                        'answer': answer,
                        'context': context[:]
                    })

                    # context += f'{answer}. '
                    context += [answer]

