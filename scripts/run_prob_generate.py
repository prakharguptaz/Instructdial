from transformers import pipeline
import torch
import argparse
import json
import logging
import numpy as np
import os
import random
import math
import copy
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict

from tqdm import tqdm
from tqdm import trange
import jsonlines

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartForConditionalGeneration, BartConfig


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

def get_json_lines(inp_file):
    lines = []
    with jsonlines.open(inp_file) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str) ##"yuchenlin/BART0pp"
    parser.add_argument("--output_folder", default='outputs/', type=str)   
    parser.add_argument("--output_prefix", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--nosample', action='store_false')

    return parser.parse_args()




# generator = pipeline("text-generation", model="yuchenlin/BART0pp")
# output = generator(
#     "In this course, we will teach you how to",
#     max_length=30,
#     num_return_sequences=2,
# )

# print(output)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_outputs_withprobs(args, model, tokenizer, batch):
    # import pdb;pdb.set_trace()
    texts = [x["prompt"] for x in batch]
    encoding = tokenizer(texts, return_tensors='pt', truncation=True, padding="max_length", max_length=1024).to(model.device)
    if 'candidates' not in batch[0]:
        print('no candidates')
        exit(0)
    label_list_str = batch[0]['candidates']
    label_list = [tokenizer.encode(x)[0] for x in label_list_str]
    with torch.no_grad():
        # import pdb;pdb.set_trace()
        input_ids = encoding['input_ids']
        # generated_texts = tokenizer.batch_decode(torch.argmax(model(input_ids = input_ids)[0], axis=-1), skip_special_tokens=True)
        # print(generated_texts)
        # print(model(encoding))
        if args.nosample:
            generated = model.generate(**encoding,max_length=60,do_sample=False, return_dict_in_generate=True, output_scores =True)
        else:
            generated = model.generate(**encoding,max_length=60,do_sample=True,top_p=0.9,top_k=0)

        generated_scores = generated['scores'][0]
        max_tokens = torch.argmax(generated_scores,dim=1)
        norm_scores = torch.softmax(generated_scores,dim=1)
        label_probs_list = []
        for label in label_list:
            probs = torch.index_select(norm_scores,1,torch.tensor([label]).cuda())[:,0].tolist()
            label_probs_list.append(probs)

        sum_prob_list = [sum(row[i] for row in label_probs_list) for i in range(len(label_probs_list[0]))]
        balanced_label_probs_list = [[label_probs_list[r][c]/sum_prob_list[c] for c in range(len(label_probs_list[r]))] for r in range(len(label_probs_list))]

    generated_texts = tokenizer.batch_decode(generated['sequences'], skip_special_tokens=True)
    transpose_balanced_label_probs_list = [list(i) for i in zip(*balanced_label_probs_list)]

    return generated_texts, transpose_balanced_label_probs_list

def run_generation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # config = BartConfig.from_pretrained(args.model)
    # config.output_past = True
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model, config=config)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model = model.to(device)
    model_name_string = args.model
    if model_name_string[-1] =='/':
        model_name_string = model_name_string[:-1]
    if 'checkpoint' in model_name_string:
        model_name_string = model_name_string.split('/')[-2] + '--' + model_name_string.split('/')[-1]
    else:
        model_name_string = model_name_string.split('/')[-1]
    if args.nosample:
        model_name_string = model_name_string + '-nosample-' 

    data_json_lines = get_json_lines(args.input_file)
    
    batch_chunks = chunks(data_json_lines, args.batch_size)
    input_file_name = args.input_file.split('/')[-1]

    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, args.output_prefix +"_"+model_name_string+ "-" + input_file_name), 'w') as outfile:
        for b, batch in tqdm(enumerate(batch_chunks), total=len(data_json_lines)//args.batch_size):
            # import pdb;pdb.set_trace()
            # print(batch)
            outputs, probs = get_outputs_withprobs(args, model, tokenizer, batch)
            # print(outputs)
            for d, dp in enumerate(batch):
                batch[d]['output'] = outputs[d]
                batch[d]['class_probability'] = probs[d]
                if b%100==0:
                    print(batch[d]['output'], batch[d]['class_probability'])
                json.dump(batch[d], outfile)
                outfile.write('\n')
                outfile.flush()

    # texts =     ["In this course, we will teach you how to", "Is this review positive or negative? Review: Best cast iron skillet you will every buy."]

    # print(generated_texts)




if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)

    run_generation(args)


# python run_generate.py --output_prefix t1 --input_file text2textfiles/traindata_tasks1k-dev.json --model tmp/tst-train2/checkpoint-2400/
