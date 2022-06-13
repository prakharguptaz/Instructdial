import os
import json
from data_utils.data_reader import Dataset
import string
import re


class SnipsDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.examples = []
        self.intent_classes = set()
        self.slot_classes = set()

        # Could not find official splits
        if split != 'train':
            return

        with open(os.path.join(data_path, 'benchmark_data.json')) as f:
            data = json.load(f)

        domains = data["domains"]

        for domain_dict in domains:
            intents = domain_dict["intents"]
            for intent_dict in intents:
                intent = intent_dict["benchmark"]["Snips"]["original_intent_name"]
                self.intent_classes.add(intent)
                queries = intent_dict["queries"]
                for query_dict in queries:
                    query_text = query_dict["text"]
                    slots_data = query_dict["results_per_service"]["Snips"]["slots"]

                    query_slots = query_text.translate(str.maketrans('', '', string.punctuation))

                    for slot in slots_data:
                        if len(slot['value']) > 0:
                            query_slots = query_slots.replace(slot['value'],
                                                              'SLOT_{}_{}'.format(len(slot['value'].split()),
                                                                                  slot['name']))

                    slots = []
                    for w in query_slots.split():
                        if w.startswith('SLOT_'):
                            m = re.match(r'SLOT_([0-9]+)_(.*)', w)
                            n, field = m.group(1), m.group(2)
                            slots.append('B-{}'.format(field))
                            for i in range(int(n) - 1):
                                slots.append('I-{}'.format(field))
                                self.slot_classes.add('I-{}'.format(field))
                        else:
                            slots.append('O')

                    slots = ' '.join(slots)

                    self.examples.append({
                        'response': query_text,
                        'slots': slots,
                        'intent_label': intent,
                        'ind': len(self.examples)
                    })

        self.slot_classes = list(self.slot_classes)
        self.intent_classes = list(self.intent_classes)
