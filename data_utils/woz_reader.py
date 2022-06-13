import os
from data_utils.data_reader import Dataset
import json
from collections import defaultdict


def update_belief_state(usr_dict, prev_bs_dict, prev_bs_name_list):
    res_bs_dict, res_bs_name_list = prev_bs_dict, prev_bs_name_list
    curr_bs_state = usr_dict['belief_state']
    for item in curr_bs_state:
        if item['act'] == 'inform':  # only care about inform act
            for pair in item['slots']:
                slot_name, value = pair
                if slot_name not in res_bs_name_list:
                    res_bs_name_list.append(slot_name)
                res_bs_dict[slot_name] = value
    if len(res_bs_name_list) == 0:
        res_text, res_dx_text = '', ''
    else:
        res_text = '[restaurant] '
        res_dx_text = '[restaurant] '
        for name in res_bs_name_list:
            value = res_bs_dict[name]
            res_text += name + ' ' + value + ' '
            res_dx_text += name + ' '
        res_text = res_text.strip().strip(' , ').strip()
        res_dx_text = res_dx_text.strip().strip(' , ').strip()
    return res_text, res_dx_text, res_bs_dict, res_bs_name_list


def zip_sess_list(sess_list):
    turn_num = len(sess_list)
    assert sess_list[0]["system_transcript"] == ''
    if turn_num == 1:
        raise Exception()
    turn_list = []
    for idx in range(turn_num - 1):
        curr_turn_dict = sess_list[idx]
        system_uttr = sess_list[idx + 1]['system_transcript']
        turn_list.append((curr_turn_dict, system_uttr))
    return turn_list


def process_session(sess_list):
    turn_num = len(sess_list)
    res_dict = {'dataset': 'WOZ',
                'dialogue_session': []}
    for idx in range(turn_num):
        if idx == 0:
            bs_dict, bs_name_list = {}, []
        one_usr_dict, one_system_uttr = sess_list[idx]
        one_usr_uttr = one_usr_dict['transcript']
        one_usr_bs, one_usr_bsdx, bs_dict, bs_name_list = \
            update_belief_state(one_usr_dict, bs_dict, bs_name_list)

        one_turn_dict = {'turn_num': idx}
        one_turn_dict['user'] = one_usr_uttr
        one_turn_dict['resp'] = one_system_uttr
        one_turn_dict['turn_domain'] = ['[restaurant]']
        one_turn_dict['bspn'] = one_usr_bs
        one_turn_dict['bsdx'] = one_usr_bsdx
        one_turn_dict['aspn'] = ''
        res_dict['dialogue_session'].append(one_turn_dict)
    return res_dict


def process_file(in_f):
    with open(in_f) as f:
        data = json.load(f)
    res_list = []
    for item in data:
        one_sess = zip_sess_list(item['dialogue'])
        if len(one_sess) == 0:
            continue
        one_res_dict = process_session(one_sess)
        res_list.append(one_res_dict)
    print(len(res_list), len(data))
    return res_list


class WozDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.examples = []
        self.slot_classes = set()
        self.act_classes = set()

        with open(os.path.join(data_path, 'woz_{}_en.json'.format(split))) as f:
            data = json.load(f)

        for dialogue in data:
            turns = dialogue['dialogue']
            context = []

            for i, turn in enumerate(turns):
                slot_data = {s[1]: s[0] for s in turn['turn_label']}
                slots = turn['asr'][0][0].lower().split()

                for j, word in enumerate(slots):
                    if word in slot_data:
                        slots[j] = 'B-{}'.format(slot_data[word])
                    else:
                        slots[j] = 'O'

                acts = defaultdict(list)
                if i > 1:
                    for info in turns[i - 1]['belief_state']:
                        act = info['act']
                        act_slots = info['slots']
                        for slot in act_slots:
                            acts[act].append(slot[0])

                response = turn['system_transcript']

                belief_str = []
                for bs in turn['belief_state']:
                    for slot in bs['slots']:
                        belief_str.append('{} : {}'.format(slot[0], slot[1]))
                belief_str = ' , '.join(belief_str)

                if len(response) > 0:
                    self.examples.append({'context': context[:],
                                          'response': response,
                                          'slots': ' '.join(slots),
                                          'state': belief_str,
                                          'act_classes': json.dumps(acts),
                                          'conv_id': dialogue['dialogue_idx']})
                    context.append(response)
                usr = turn['asr'][0][0]
                context.append(usr)
                self.act_classes.update(set(acts))
                self.slot_classes.update(set(slot_data.values()))

        self.slot_classes = list(self.slot_classes)
        self.act_classes = list(self.act_classes)
