import os
import copy
import json
import numpy as np
import os
import pickle
import torch

from collections import defaultdict
from tqdm import tqdm
from data_utils.data_reader import Dataset


def filter_dataset(dataset,
                   data_type="happy", # happy, unhappy, multitask
                   domain=None,
                   task=None,
                   exclude=False,
                   percentage=1.0,
                   train=True):
    """
    Split the dataset according to the criteria

    - data_type:
        - happy: Only the happy dialogs
        - unhappy: Only the happy + unhappy dialogs (no multitask)
        - multitask: All the dialogs

    - domain:
        - Requirements:
            - task should be None
            - data_type should be happy/unhappy
        - If exclude = True
            - Exclude dialog of this domain
        - If exclude = False
            - Include ONLY dialogs of this domain

    - task:
        - Requirements:
            - domain should be None
            - data_type should be happy/unhappy
        - If exclude = True
            - Exclude dialog of this domain
        - If exclude = False
            - Include ONLY dialogs of this domain

    - percentage:
        - Take only a certain percentage of the available data (after filters)
        - If train = True
            - Take the first [percentage]% of the data
        - If train = False:
            - Take the last [percentage]% of the data
    """
    examples = dataset.examples

    # Filter based on happy/unhappy/multitask
    if data_type == "happy":
        examples = [ex for ex in examples if ex.get("happy")]
    elif data_type == "unhappy":
        examples = [ex for ex in examples if not ex.get("multitask")]

    # Filter based on domain
    if domain is not None:
        assert data_type in ["happy", "unhappy"], "For zero-shot experiments, can only use single-task data"
        assert task is None, "Can filter by domain OR task, not both"

        if exclude:
            examples = [ex for ex in examples if ex["domains"][0] != domain]
        else:
            examples = [ex for ex in examples if ex["domains"][0] == domain]

    # Filter based on task
    if task is not None:
        assert data_type in ["happy", "unhappy"], "For zero-shot experiments, can only use single-task data"
        assert domain is None, "Can filter by domain OR task, not both"

        if exclude:
            examples = [ex for ex in examples if ex["tasks"][0] != task]
        else:
            examples = [ex for ex in examples if ex["tasks"][0] == task]

    # Split based on percentage
    all_dialog_ids = sorted(list(set([ex['dialog_id'] for ex in examples])))
    if train:
        selected_ids = all_dialog_ids[:int(len(all_dialog_ids)*percentage)]
    else:
        selected_ids = all_dialog_ids[-int(len(all_dialog_ids)*percentage):]

    selected_ids = set(selected_ids)
    examples = [ex for ex in examples if ex['dialog_id'] in  selected_ids]

    # Filter out only the relevant keys for each example (so that DataLoader doesn't complain)
    keys = ["input_ids", "attention_mask", "token_type_ids", "action", "tasks", "history", "response"]
    examples = [{k:v for k,v in ex.items() if k in keys} for ex in examples]
    for ex in examples:
        ex["tasks"] = ex["tasks"][0]

    # Return new dataset
    new_dataset = copy.deepcopy(dataset)
    new_dataset.examples = examples
    return new_dataset


class STARDataset(Dataset):
    def __init__(self,
                 data_path,
                 max_seq_length,
                 vocab_file_name,
                 train=True):
        # Read all of the JSON files in the data directory
        conversations = [
            json.load(open(data_path + fn)) for fn in os.listdir(data_path)
        ]

        # Iterate over the conversations and get (1) the dialogs and (2) the 
        # actions for all wizard turns.
        self.examples = []
        self.action_label_to_id = {}
        for conv in tqdm(conversations):
            # History (so far) for this dialog
            history = ""
            for i,utt in enumerate(conv['Events']):
                # Process and concatenate to history
                if utt['Agent'] in ['User', 'Wizard', 'KnowledgeBase']:
                    utt_text = ""

                    # If there's text, just use it directly
                    if utt['Action'] in ['pick_suggestion', 'utter']:
                        utt_text = utt['Text']

                    # If it's a knowledge base query, format it as a string
                    if utt['Action'] == 'query':
                        utt_text = "[QUERY] "
                        for constraint in utt['Constraints']:
                            key = list(constraint.keys())[0]
                            val = constraint[key]

                            utt_text += "{} = {} ; ".format(key, val)

                    # If it's a knowledge base item, format it as a string
                    if utt['Action'] == 'return_item':
                        utt_text = "[RESULT] "
                        if 'Item' not in utt:
                            utt_text += "NO RESULT"
                        else:
                            for key,val in utt['Item'].items():
                                utt_text += "{} = {} ; ".format(key, val)


                # NOTE: Ground truth action labels only exist when wizard picks suggestion. 
                # We skip all custom utterances for action prediction.
                if utt['Agent'] == 'Wizard' and utt['Action'] in ['query', 'pick_suggestion']:
                    # Tokenize history
                    processed_history = ' '.join(history.strip().split()[:-1])

                    # Convert action label to id
                    query_label = 'query'
                    if 'ActionLabel' not in utt:
                        query_check = 'Check' in [e['RequestType'][1:-1] for e in utt['Constraints'] if 'RequestType' in e]
                        query_book = 'Book' in [e['RequestType'][1:-1] for e in utt['Constraints'] if 'RequestType' in e]
                        # In case of a bug, if both book and check are on - we treat it as a check.
                        if query_check:
                            query_label = 'query_check' 
                        elif query_book:
                            query_label = 'query_book' 

                    action_label = utt['ActionLabel'] if 'ActionLabel' in utt else query_label
                    if action_label not in self.action_label_to_id:
                        self.action_label_to_id[action_label] = len(self.action_label_to_id)
                    action_label_id = self.action_label_to_id[action_label]

                    # Include metadata 
                    domains = conv['Scenario']['Domains']
                    tasks = [e['Task'] for e in conv['Scenario']['WizardCapabilities']]
                    happy = conv['Scenario']['Happy']
                    multitask = conv['Scenario']['MultiTask']

                    # Add to data
                    self.examples.append({
                        "action": action_label_id,
                        "dialog_id": conv['DialogueID'],
                        "domains": domains,
                        "tasks": tasks,
                        "happy": happy,
                        "multitask": multitask,
                        "orig_history": processed_history,
                        "orig_action": action_label,
                        "response": utt_text,
                    })


                # Process and concatenate to history
                if utt['Agent'] in ['User', 'Wizard', 'KnowledgeBase']:
                    if utt_text != "":
                        history += "[{}] {} [SEP] ".format(utt['Agent'], utt_text.strip())
        if train:
            filter_dataset(self, data_type="multitask", percentage=0.8, train=True)
        else:
            filter_dataset(self, data_type="multitask", percentage=0.2, train=False)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
