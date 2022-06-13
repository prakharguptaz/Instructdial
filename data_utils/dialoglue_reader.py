import csv
import logging
import json
import numpy as np
import os
import pickle
import re
from collections import defaultdict
from tokenizers import BertWordPieceTokenizer
# from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict

from constants import SPECIAL_TOKENS

from data_utils.data_reader import Dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

class IntentDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 # tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        # self.split = os.path.basename(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "categories.json")
        intent_names = json.load(open(intent_vocab_path))
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}
        self.examples = []
        reader = csv.reader(open(data_path))
        next(reader, None)
        out = []
        # print(self.intent_idx_to_label)
        self.intent_classes = list(self.intent_idx_to_label.values())
        self.intent_classes = [x.replace('_',' ') for x in self.intent_classes]
        for utt, intent in tqdm(reader):
            self.examples.append({
                "response": utt,
                "intent_label": intent,#self.intent_label_to_idx[intent],
                "index": len(self.examples),
            })

        self.idx=0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def get_next(self):
        if self.idx>=len(self.examples):
            return None
        dp = self.examples[self.idx]
        self.idx+=1

        return dp

slots_map = {"time": "time", "people": "number of people", "first_name": "first name", "last_name": "last name", "date": "date", 'pickup_time': 'pickup time', 'to_location': 'going to', 'date': 'date', 'pickup_date': 'pickup date', 'dropoff_date': 'dropoff date', 'subcategory': 'subcategory', 'leaving_date': 'date', 'city_of_event': 'city of the event', 'pickup_city': 'pickup city', 'area': 'area', 'from_location': 'leaving from', 'visit_date': 'visit date'}


class SlotDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 # tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        self.split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.txt")
        slot_names = json.load(open(slot_vocab_path))
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        self.slot_classes = list(self.slot_idx_to_label.values())
        self.slot_classes = [x.replace('_',' ') for x in self.slot_classes]
        # Process data
        # self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_slots_cached".format(self.split, vocab_file_name))
        texts = []
        slotss = []
        self.examples = []
        data = json.load(open(data_path))
        for example in tqdm(data):
            text, slots = self.parse_example(example)
            context=''
            # if 'restaurant8k' in data_path:
            if 'context' in example and 'requestedSlots' in example['context']:
                requested_slots = example['context']['requestedSlots']
                context = ["What is the " + ", ".join([slots_map.get(e) for e in requested_slots if e in slots_map]) + "?" ]
                if context[0] == "What is the ?":
                    context = ''
            # if len(text)<7:
            #     print(text)
            #     print(context)
            #     import pdb; pdb.set_trace()
            texts.append(text)
            slotss.append(slots)
            # encoded = tokenizer.encode(text)
            # encoded_slot_labels = self.encode_token_labels([text], [slots],
            #                                                # len(encoded.ids),
            #                                                # tokenizer,
            #                                                self.slot_label_to_idx,
            #                                                max_seq_length)
            self.examples.append({
                "context":context, 
                "text": text,
                "example": example,
                "slots": slots,
                # "slot_labels": encoded_slot_labels[-max_seq_length:],
                "index": len(self.examples),
            })


    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            # encoded_length,
                            # tokenizer,
                            slot_map,
                            max_length) -> np.array:

        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))

        # encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            # import pdb;pdb.set_trace()
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def parse_example(self, example):
        text = example['userInput']['text']

        # Create slots dictionary
        word_to_slot = {}
        for label in example.get('labels', []):
            slot = label['slot']
            start = label['valueSpan'].get('startIndex', 0)
            end = label['valueSpan'].get('endIndex', -1)

            for word in text[start:end].split():
                word_to_slot[word] = slot

        # Add context if it's there
        # if 'context' in example:
        #     for req in example['context'].get('requestedSlots', []):
        #         text = req + " " + text

        # Create slots list
        slots = []
        cur = None
        # if len(word_to_slot)!=0:
        #     print(word_to_slot)
        #     import pdb;pdb.set_trace()
        for word in text.split():
            word = re.sub('[\W_]+', '', word)
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

        # print(text, " ".join(slots))

        return text, " ".join(slots)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class TOPDataset(Dataset):
    def __init__(self, data_path, config):
        data_path = os.path.join('.', data_path, 'top')
        # Slot categories
        slot_vocab_path = os.path.join(data_path, config['slot_vocab'])
        slot_names = [e.strip() for e in open(slot_vocab_path).readlines()]
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Intent categories
        intent_vocab_path = os.path.join(data_path,  config['intent_vocab'])
        intent_names = [e.strip() for e in open(intent_vocab_path).readlines()]
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.examples = []

        data_file = os.path.join(data_path,  config['{}_data_path'.format(config['split'])])
        data = [e.strip() for e in open(data_file).readlines()]
        for example in tqdm(data):
            example, intent = example.split(" <=> ")
            text = " ".join([e.split(":")[0] for e in example.split()])
            slots = " ".join([e.split(":")[1] for e in example.split()])

            self.examples.append({
                "text": text,
                "slots": slots,
                "intent_label": intent,
                "index": len(self.examples),
            })

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:

        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))

        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
