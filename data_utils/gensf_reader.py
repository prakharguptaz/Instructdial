

import csv
import logging
import json
import numpy as np
import os
import pickle

from tqdm import tqdm

from data_utils.data_reader import Dataset
class GensfDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 split: str,
                 domain='all'):
        texts = []
        slotss = []
        self.examples = []
        
        if domain == 'all':
            domains = ['Buses_1', 'Events_1', 'Homes_1', 'RentalCars_1']
        else:
            domains = [domain]

        if split == 'train' or split == 'val':
            file = 'train_2.json' # 1/4 examples
        elif split == 'test':
            file = 'test.json'
        
        for domain in domains:
            file_path = f'{data_path}/dstc8/{domain}/{file}'
            data = json.load(open(file_path))
            for example in tqdm(data):
                text, slots, slot_to_word = self.parse_example(example)
                texts.append(text)
                slotss.append(slots)

                self.examples.append({
                    "text": text,
                    "slots": slots,
                    "index": len(self.examples),
                    'domain': domain,
                    'slot_to_word':slot_to_word
                })
            
        self.split = split
        
        num_examples = len(self.examples)
        if split == 'train': 
            self.exampels = self.examples#[:-num_examples // 10]
        elif split == 'val':
            self.examples = self.examples[-num_examples//10:]

    def parse_example(self, example):
        text = example['userInput']['text']
        # Create slots dictionary
        word_to_slot = {}
        slot_to_word = {}
        for label in example.get('labels', []):
            slot = label['slot']
            start = label['valueSpan'].get('startIndex', 0)
            end = label['valueSpan'].get('endIndex', -1)

            for word in text[start:end].split():
                word_to_slot[word] = slot

            slot_to_word[slot] = [text[start:end]]



        # Add context if it's there
        # if 'context' in example:
        #     for req in example['context'].get('requestedSlots', []):
        #         text = req + " " + text

        # Create slots list
        slots = []
        cur = None
        for word in text.split():
            if word in word_to_slot:
                slot = word_to_slot[word]
                if cur is not None and slot == cur:
                    slots.append("I-" + slot)
                else:
                    slots.append("B-" + slot)
                    cur = slot
            else:
                slots.append("O")
                cur = None

        return text, " ".join(slots), slot_to_word

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

