import os
from data_utils.data_reader import Dataset
import json


class CasinoDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.examples = []
        self.strategy_classes = set([])

        with open(os.path.join(data_path, 'casino_{}.json'.format(split))) as f:
            data = json.load(f)

            for chat in data:
                annotated_chat = chat['annotations']
                if len(annotated_chat) == 0:
                    continue
                context = [annotated_chat[0][0]]
                for utterance in annotated_chat[1:]:
                    utterance_text, strategy = utterance[0], utterance[1]
                    strategy = strategy.split(',')
                    self.examples.append({"context": context[:],
                                          "response": utterance_text,
                                          "strategy": strategy})
                    self.strategy_classes = self.strategy_classes.union(set(strategy))
                    context.append(utterance_text)

        self.strategy_classes = list(self.strategy_classes)
