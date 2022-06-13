import random
import nltk
from nltk.tokenize import RegexpTokenizer
import torch

def read_dataset(path):
    '''
    - A line of sentence x -> Output a of sentence x
    - A line of 2 sentences separated by tab -> a tuple of (c,r)
    '''
    arr = []
    with open(path) as f:
        for line in f:
            sents = [ x.strip() for x in line.split('\t') ]
            if len(sents) == 1:
                arr.append(sents[0])
            else:
                arr.append(sents)
        f.close()
    return arr

def apply_word_order(tokens):
    random.shuffle(tokens)
    return " ".join(tokens)

def apply_word_drop(tokens):
    drop_threshold = random.random() / 2
    arr = []
    for token in tokens:
        val = random.random()
        if val < drop_threshold :
            continue
        arr.append(token)
    # keep on running until there is at least one token
    if len(arr) == 0 or len(arr) == len(tokens):
        return apply_word_drop(tokens)
    return " ".join(arr)

def apply_word_repeat(tokens):
    arr = []
    for token in tokens:
        arr.append(token)

        # check whether or not to duplicate
        if random.random() < 0.9:
            continue

        # get how many words to duplicate
        max_word = min(len(arr), 3)
        word_probab = [1]
        if max_word == 2:
            word_probab = [0.75, 1]
        elif max_word == 3:
            word_probab = [0.7, 0.9, 1]

        threshold = random.random()
        for idx, val in enumerate(word_probab):
            if threshold < val:
                break
        max_word = idx+1

        # get how many time to duplicate
        threshold2 = random.random()
        time_probab = [0.8, 0.95, 1]
        for idx, val in enumerate(time_probab):
            if threshold2 < val:
                break
        time = idx+1

        arr += (arr[-max_word:]) * time

    if (len(arr) == len(tokens)):
        return apply_word_repeat(tokens)
    return " ".join(arr)

def apply_word_reverse(tokens):
    tokens = tokens[::-1]
    return " ".join(tokens)

def apply_retain_noun(tokens):
    arr = []
    pos_tags = nltk.pos_tag(tokens)
    for token, pos in pos_tags:
        if pos in ['NN', 'NNP']:
            arr.append(token)
    return " ".join(arr)

def apply_remove_puntuation(tokens):
    if tokens[-1] in ['.', '!', '?', '...', '!!!', '!!', '..', ',', ',,']:
        return " ".join(tokens[:-1])
    return " ".join(tokens)

def apply_remove_stopwords(tokens):
    #stopwords = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'])
    #stopwords = set(['a', 'an', 'the', 'have', 'has', 'had', 'be', 'is', 'am', 'are', 'was', 'were', 'been', 'will', 'can', 'may', 'would', 'could', 'might'])
    stopwords = set(['a', 'an', 'the', 've', '\'', 's', 'll', 'd', 're', 'm', 't'])
    arr = []
    for token in tokens:
        if token not in stopwords:
            arr.append(token)
    return " ".join(arr)

def apply_simplify_response(tokens):
    tags = nltk.pos_tag(tokens)
    arr = []
    for word, tag in tags:
        if tag in ['NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBZ', 'VBN', 'VBG', 'VBD', 'IN']:
            arr.append(word)
    return " ".join(arr)


def encode_truncate (tokenizer, s1, s2, ctx_token_len=25, res_token_len=25):
    '''
    This function is to truncate s1 and s2 before concatenating.
    s1 is truncated at the start, whereas s2 is at the end of tokens.
    '''
    s1_tokens = tokenizer.tokenize(s1)
    s2_tokens = tokenizer.tokenize(s2)
    s1_length = len(s1_tokens)
    s2_length = len(s2_tokens)

    # make the s1 and s2 shorter together
    if s1_length + s2_length > ctx_token_len + res_token_len:
        if s1_length > ctx_token_len:
            s1_tokens = s1_tokens[-ctx_token_len:]
            s1 = " ".join(s1_tokens)
        if s2_length > res_token_len:
            s2_tokens = s2_tokens[:res_token_len]
            s2 = " ".join(s2_tokens)

    max_length = ctx_token_len + res_token_len
    input_dict = tokenizer.encode_plus(s1, s2, max_length=max_length, truncation=True)

    # process
    input_ids = input_dict['input_ids']
    token_type_ids = input_dict['token_type_ids']
    input_len = len(input_ids)
    padding_len = max_length - len(input_ids)

    # add the padding
    input_ids += [0] * padding_len
    token_type_ids += [1] * padding_len
    mask_tokens = [1] * input_len + [0] * padding_len
    pos_ids = [i for i in range(input_len)] + [0]*padding_len

    # return input_ids, token_type_ids, mask_tokens, pos_ids
    return  torch.LongTensor(input_ids), \
            torch.LongTensor(token_type_ids), \
            torch.Tensor(mask_tokens), \
            torch.LongTensor(pos_ids)
