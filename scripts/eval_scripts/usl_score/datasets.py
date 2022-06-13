from data_utils import *
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class VUPDataset(Dataset):
    def __init__(self, instances, maxlen=25):
        self.maxlen = maxlen
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def apply_negative_syntax(self, x):
        '''
            #0: Reorder
            #1: Word Drop
            #2: Word Repeat
        '''
        probabs = [0.3, 0.6, 1]
        thresh = random.random()
        for idx, val in enumerate(probabs):
            if thresh < val:
                break

        tokens = self.tokenizer.tokenize(x)
        if idx == 0:
            return apply_word_order(tokens)
        elif idx == 1:
            return apply_word_drop(tokens)
        else:
            return apply_word_repeat(tokens)

    def apply_positive_syntax(self, x):
        '''
            #0: Remove Puntuation at the end
            #1: Simplify Response
            #2: Remove Stopword
        '''
        probabs = [0.1, 0.2, 0.3, 1]
        thresh = random.random()
        for idx, val in enumerate(probabs):
            if thresh < val:
                break

        tokens = self.tokenizer.tokenize(x)
        if idx == 0:
            return apply_remove_puntuation(x)
        elif idx == 1:
            return apply_simplify_response(tokens)
        elif idx == 2:
            return apply_remove_stopwords(tokens)
        else:
            return x

    def __getitem__(self, index):
        instance = self.instances[index]
        label = 1

        # apply negative syntactic
        tokens = self.tokenizer.tokenize(instance)
        if random.random() > 0.5 and len(tokens) >= 3:
            instance = self.apply_negative_syntax(instance)
            label = 0
        else:
            instance = self.apply_positive_syntax(instance)
            label = 1

        instance = self.tokenizer.encode_plus(instance,
                                         add_special_tokens=True,
                                         max_length=self.maxlen,
                                         pad_to_max_length=True,
                                         return_tensors="pt")
        input_ids = instance['input_ids']
        token_type_ids = instance['token_type_ids']
        attention_mask = instance['attention_mask']
        return input_ids, token_type_ids, attention_mask, label

class NUPDataset(Dataset):
    def __init__(self, contexts, responses, ctx_token_len=25, res_token_len=25):
        self.ctx_token_len = ctx_token_len
        self.res_token_len = res_token_len
        self.contexts = contexts
        self.responses = responses
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._build_response_pool()

    def _build_response_pool(self):
        self.res_pool = self.responses

    def _get_fake_response(self):
        idx = random.randint(0, len(self.res_pool)-1)
        return self.res_pool[idx]

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, index):
        'Generates one sample of data'
        ctx = self.contexts[index]
        res = self.responses[index]
        label = 1

        # negative sampling
        if random.random() < 0.5:
            res = self._get_fake_response()
            label = 0

        # Encode the input
        input_ids, token_type_ids, mask_tokens, pos_ids = encode_truncate(
            self.tokenizer,
            ctx, res,
            ctx_token_len=self.ctx_token_len,
            res_token_len=self.res_token_len
        )

        return input_ids, token_type_ids, mask_tokens, pos_ids, label

class MLMDataset(Dataset):
    def __init__(self, instances, maxlen):
        self.maxlen = maxlen
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        instance = self.instances[index]

        tokens = self.tokenizer.tokenize(instance)
        instance = self.tokenizer.encode_plus(instance,
                                         add_special_tokens=True,
                                         max_length=self.maxlen,
                                         pad_to_max_length=True,
                                         return_tensors="pt")
        input_ids = instance['input_ids']
        token_type_ids = instance['token_type_ids']
        attention_mask = instance['attention_mask']

        # mask a token
        sampling_length = min(len(tokens)+2, self.maxlen)
        mask_idx = torch.LongTensor([random.randint(1,sampling_length-2)])
        label = torch.LongTensor([input_ids[0][mask_idx].item()])
        input_ids[0][mask_idx] = 103 # [MASK] token <- 103

        return input_ids, token_type_ids, attention_mask, mask_idx, label


if __name__ == "__main__":
    sents = [
        "i go to school.",
        "really? you don't like burger?"
    ]
    dataset = VUPDataset(sents)
    for x in dataset:
        print (x)
