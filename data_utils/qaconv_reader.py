from typing import List
import json
from data_utils.data_reader import Dataset


class QAConvDataset(Dataset):
    def __init__(self, data_path: str, max_seq_length=512, split='test'):

        self.examples = []
        article_json = json.load(open(f"{data_path}/article_segment.json"))

        if split == 'train':
            ques_json = json.load(open(f"{data_path}/trn.json"))
        elif split == 'dev':
            ques_json = json.load(open(f"{data_path}/val.json"))
        elif split == 'test':
            ques_json = json.load(open(f"{data_path}/tst.json"))

        
        for qa_pair in ques_json:
            context = article_json[qa_pair["article_segment_id"]]["seg_dialog"]
            context = " ".join(['{}: {}'.format(c["speaker"], c["text"].replace("\n", " ")) for c in context])

            if len(qa_pair["answers"]):
                tgt = qa_pair["answers"][0].strip() # here we only use the first potential answers
            else: # unanswerable
                tgt = "unanswerable"
            
            qg_context = context.strip()
            question = qa_pair['question'].strip()
            
            context = f'{qg_context} {question}'
            self.examples.append({
                'qg_context': qg_context,
                'question': question,
                'context': context,
                'answer': tgt
            })
        self.split = split