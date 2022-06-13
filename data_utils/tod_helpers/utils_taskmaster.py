import json
import ast
import collections
import os
import re

from .utils_function import get_input_example


def read_langs_turn(args, dials, ds_name, max_line):
    print(("Reading from {} for read_langs_turn".format(ds_name)))
    slot_classes = set([])

    data = []
    for dial in dials:
        for utterance in dial['utterances']:
            filtered_text = utterance['text']
            slots = []
            slots_dict = {}
            if 'segments' in utterance:
                for seg in utterance['segments']:
                    slot_text = seg['text']
                    slot_keys_annotations = seg['annotations']
                    for slot_key_annotation in slot_keys_annotations:
                        slot_name = slot_key_annotation['name']
                        if slot_name not in slots_dict:
                            slots_dict[slot_name] = []
                        slots_dict[slot_name].append(slot_text)

                for seg in reversed(utterance['segments']):
                    filtered_text = filtered_text[:seg['start_index']] + ' [SEP] ' + filtered_text[seg['end_index']:]
                filtered_text = re.sub(r'  +', ' ', filtered_text).strip()
                i = 0
                for word in filtered_text.split():
                    if word == '[SEP]':
                        slot_name = utterance['segments'][i]['annotations'][0]['name']
                        slots.append('B-{}'.format(slot_name))
                        slot_classes.add(slot_name)
                        for j in range(len(utterance['segments'][i]['text'].split()) - 1):
                            slots.append('I-{}'.format(slot_name))
                        i += 1
                    else:
                        slots.append('O')
            else:
                slots = 'O' * len(utterance['text'])
            data.append({
                'text': utterance['text'],
                'speaker': utterance['speaker'],
                'slots': ' '.join(slots),
                'slots_dict': slots_dict,
                'conversation_id': dial['conversation_id'],
                'instruction_id': dial['instruction_id'],
            })

    slot_classes = list(slot_classes)

    return data, slot_classes


def read_langs_dial(file_name, ontology, dialog_act, max_line=None, domain_act_flag=False):
    print(("Reading from {} for read_langs_dial".format(file_name)))

    raise NotImplementedError


def prepare_data_taskmaster(args):
    ds_name = "TaskMaster"

    example_type = args["example_type"]
    max_line = args["max_line"]

    fr_trn_id = open(os.path.join(args["data_path"], 'Taskmaster/TM-1-2019/train-dev-test/train.csv'), 'r')
    fr_dev_id = open(os.path.join(args["data_path"], 'Taskmaster/TM-1-2019/train-dev-test/dev.csv'), 'r')
    fr_trn_id = fr_trn_id.readlines()
    fr_dev_id = fr_dev_id.readlines()
    fr_trn_id = [_id.replace("\n", "").replace(",", "") for _id in fr_trn_id]
    fr_dev_id = [_id.replace("\n", "").replace(",", "") for _id in fr_dev_id]

    fr_data_woz = open(os.path.join(args["data_path"], 'Taskmaster/TM-1-2019/woz-dialogs.json'), 'r')
    fr_data_self = open(os.path.join(args["data_path"], 'Taskmaster/TM-1-2019/self-dialogs.json'), 'r')
    dials_all = json.load(fr_data_woz) + json.load(fr_data_self)

    _example_type = "dial" if "dial" in example_type else example_type
    pair_trn, slot_classes = globals()["read_langs_{}".format(_example_type)](args, dials_all, ds_name, max_line)
    pair_dev = []
    pair_tst = []

    print("Read {} pairs train from {}".format(len(pair_trn), ds_name))
    print("Read {} pairs valid from {}".format(len(pair_dev), ds_name))
    print("Read {} pairs test  from {}".format(len(pair_tst), ds_name))

    meta_data = {"num_labels": 0,
                 "slot_classes": slot_classes}

    return pair_trn, pair_dev, pair_tst, meta_data
