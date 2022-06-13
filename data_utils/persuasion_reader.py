import os
from data_utils.data_reader import Dataset
import pandas as pd
import math


class PersuasionDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.idx = 0
        self.examples = []

        if split == 'train':
            df_1 = pd.read_csv(os.path.join(data_path, 'train0.csv')).iloc[:, 1:]
            df_2 = pd.read_csv(os.path.join(data_path, 'train1.csv')).iloc[:, 1:]
            df_3 = pd.read_csv(os.path.join(data_path, 'train2.csv')).iloc[:, 1:]
            df_4 = pd.read_csv(os.path.join(data_path, 'train3.csv')).iloc[:, 1:]
            df_5 = pd.read_csv(os.path.join(data_path, 'train4.csv')).iloc[:, 1:]

        if split == 'test':
            df_1 = pd.read_csv(os.path.join(data_path, 'test0.csv')).iloc[:, 1:]
            df_2 = pd.read_csv(os.path.join(data_path, 'test1.csv')).iloc[:, 1:]
            df_3 = pd.read_csv(os.path.join(data_path, 'test2.csv')).iloc[:, 1:]
            df_4 = pd.read_csv(os.path.join(data_path, 'test3.csv')).iloc[:, 1:]
            df_5 = pd.read_csv(os.path.join(data_path, 'test4.csv')).iloc[:, 1:]

        df = pd.concat([df_1, df_2, df_3, df_4, df_5])
        df.reset_index(inplace=True)
        examples = df.to_dict('index')
        for index, row in examples.items():
            self.examples.append({
                "context": row['history'],
                "response": row['Unit'],
                "strategy": [row['er_label_1']]
            })

        self.strategy_classes = df['er_label_1'].unique().tolist()

        """
        df = pd.read_excel(os.path.join(data_path, '300_dialog.xlsx')).iloc[:, 1:]
        ids = df['B2'].unique()
        dialogs = [list(df[df['B2'] == id].to_dict('index').values()) for id in ids]

        for dialog in dialogs:
            context = [dialog[0]['Unit']]
            for utterance in dialog[1:]:
                if type(utterance['er_label_1']) == str:
                    self.examples.append(
                        {"context": context[:],
                         "response": utterance['Unit'],
                         "strategy": [utterance['er_label_1']]
                         }
                    )
                    self.strategy_classes.add(utterance['er_label_1'])
                context.append(utterance['Unit'])

        self.strategy_classes = list(self.strategy_classes)
        """
