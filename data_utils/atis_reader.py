import os
from data_utils.data_reader import Dataset


class AtisDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.idx = 0
        self.examples = []

        slots_dict = self.read("atis.dict.slots.csv", dict=True)
        intents_dict = self.read("atis.dict.intent.csv", dict=True)
        vocab_dict = self.read("atis.dict.vocab.csv", dict=True)
        train_slots = self.read("atis.{}.slots.csv".format(split))
        train_intents = self.read("atis.{}.intent.csv".format(split))
        train_query = self.read("atis.{}.query.csv".format(split))
        self.slot_classes = list(slots_dict.values())
        self.intent_classes =  list(intents_dict.values())

        for slots, intent, query in zip(train_slots, train_intents, train_query):
            query = query[1: len(query)-1]
            slots = slots[1: len(slots)-1]
            slots_out = ' '.join([slots_dict[int(s)] for s in slots])
            intent_out = intents_dict[int(intent[0])]
            query_out = ' '.join([vocab_dict[int(q)] for q in query])
            self.examples.append({
                'response': query_out,
                'intent_label': intent_out,
                'slots': slots_out,
                'ind': len(self.examples)
            })

    def read(self, path, dict=False):
        with open(os.path.join(self.data_path, path)) as f:
            lines = f.readlines()
        if dict:
            return {i: lines[i].strip() for i in range(len(lines))}
        else:
            return [l.strip().split() for l in lines]
