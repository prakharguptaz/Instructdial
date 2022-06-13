import json
import os
from data_utils.data_reader import Dataset
import re
from collections import defaultdict


def parse_row(row):
    text, speaker, session_id = row[4], row[3], row[0]
    dialog_acts = [da for da in row[5:] if len(da.strip()) > 0]

    slots = defaultdict(dict)
    for acts in dialog_acts:
        m = re.match(r'([A-Za-z_]+)(\(.*\))', acts)
        intent = m.group(1)
        slots_data = m.group(2)[1:len(m.group(2)) - 1]

        if len(slots_data) > 0:
            slots_data = slots_data.split(';')
            for slot_data in slots_data:
                m1 = re.match(r'([A-Za-z_]+)=\{(.*)\}', slot_data)
                m2 = re.match(r'([A-Za-z_]+)=(.+)', slot_data)

                key, value = None, None
                if m1:
                    key, value = m1.group(1), m1.group(2)
                elif m2:
                    key, value = m2.group(1), m2.group(2)
                else:
                    key = slot_data

                if value:
                    slots[intent][key] = value
                else:
                    slots[intent][key] = ''
        else:
            slots[intent] = {}

    acts = json.dumps(slots)

    slot_str = text
    slots_data = []
    for act, act_data in slots.items():
        for slot_name, value in act_data.items():
            if len(value) > 0:
                slots_data.append((slot_name, value))

    for slot_name, value in slots_data:
        n = len(value.split())
        to_replace = ['B-{}'.format(slot_name)]
        for word in range(n - 1):
            to_replace.append('I-{}'.format(slot_name))
        slot_str = slot_str.replace(value, ' '.join(to_replace))

    slot_str = slot_str.split()
    for i, word in enumerate(slot_str):
        if not word.startswith('B-') and not word.startswith('I-'):
            slot_str[i] = 'O'

    slot_classes = [s[0] for s in slots_data]
    return slot_classes, {'context': [text],
                          'response': speaker,
                          'slots': ' '.join(slot_str),
                          'acts': acts,
                          'conv_id': session_id}


class Me2eDataset(Dataset):
    def __init__(self, data_path,  split='train'):
        # Could not find official train/test splits for this task
        self.idx = 0

        self.examples = []
        self.slot_classes = set()
        self.act_classes = set()

        if split != 'train':
            return

        paths = [os.path.join(data_path, 'movie_all.tsv'), os.path.join(data_path, 'restaurant_all.tsv')]
        for path in paths:
            with open(path) as f:
                data_reader = [l.split('\t') for l in f.readlines()][1:]

            conv_id = '1'
            context = []
            for row in data_reader:
                text, speaker, session_id = row[4], row[3], row[0]
                dialog_acts = [da for da in row[5:] if len(da.strip()) > 0]

                if session_id != conv_id:
                    context = []
                    conv_id = session_id

                slots = defaultdict(dict)
                for acts in dialog_acts:
                    m = re.match(r'([A-Za-z_]+)(\(.*\))', acts)
                    intent = m.group(1)
                    slots_data = m.group(2)[1:len(m.group(2)) - 1]

                    if len(slots_data) > 0:
                        slots_data = slots_data.split(';')
                        for slot_data in slots_data:
                            m1 = re.match(r'([A-Za-z_]+)=\{(.*)\}', slot_data)
                            m2 = re.match(r'([A-Za-z_]+)=(.+)', slot_data)

                            key, value = None, None
                            if m1:
                                key, value = m1.group(1), m1.group(2)
                            elif m2:
                                key, value = m2.group(1), m2.group(2)
                            else:
                                key = slot_data

                            if value:
                                slots[intent][key] = value
                            else:
                                slots[intent][key] = ''
                    else:
                        slots[intent] = {}

                slot_str = text
                slots_data = []
                for act, act_data in slots.items():
                    for slot_name, value in act_data.items():
                        if len(value) > 0:
                            slots_data.append((slot_name, value))

                for slot_name, value in slots_data:
                    value = value.strip()
                    n = len(value.split())
                    to_replace = ['B-{}'.format(slot_name)]
                    for word in range(n - 1):
                        to_replace.append('I-{}'.format(slot_name))
                    restring = ' '.join(to_replace)
                    slot_str = slot_str.replace(f'{value}', restring)
                    # if slot_str.endswith(value):
                    #     slot_str = slot_str.replace(f' {value}', f" {restring}")
                    # elif slot_str.endswith(value):
                    #     slot_str = slot_str.replace(f'{value} ', f"{restring} ")
                    # else:
                    #     slot_str = slot_str.replace(f' {value} ', f" {restring} ")

                # if 'B-numberofpeople45pm' in slot_str: # todo

                slot_str = slot_str.split()
                for i, word in enumerate(slot_str):
                    if not word.startswith('B-') and not word.startswith('I-'):
                        slot_str[i] = 'O'

                acts = slots
                act_classes = list(acts.keys())
                if speaker == 'agent':
                    acts = json.dumps(acts)

                    self.examples.append({'context': context[:],
                                          'response': text,
                                          'slots': ' '.join(slot_str),
                                          'acts': json.dumps(acts),
                                          'act_classes': act_classes,
                                          'conv_id': session_id})

                context.append(text)

                slot_classes = set([s[0] for s in slots_data])
                self.slot_classes.update(slot_classes)
                self.act_classes.update(set(act_classes))

        self.slot_classes = list(self.slot_classes)
        self.act_classes = list(self.act_classes)
