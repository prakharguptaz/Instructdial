import requests
import sys
import json
from dialog_parser import get_requests
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
    parser.add_argument("--port", type=str, default="10234")
    parser.add_argument('--nosample', action='store_true')

    return parser.parse_args()



# server = sys.argv[1]
# if server.isdigit():
#     server = "http://localhost:" + args.port
# inputfile = sys.argv[2]
# outputfile = sys.argv[3]
# if(len(sys.argv) > 4):
#     maxitems = int(sys.argv[4])
# else:
#     maxitems = 0

# with open(inputfile) as infile:
#     data = json.load(infile)

# if isinstance(data,dict):
#     data = get_requests(data)

# if(maxitems > 0):    
#     data = data[:maxitems]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_responsesfromdict(dp):
    if dp['task']=='eval_binary':
        dp['agent_name'] = "test"
        usefultext = dp['input'].split('[CONTEXT] ')[1].split(' [ENDOFDIALOGUE]')[0]
        response = usefultext.split(' [RESPONSE] ')[1]
        ctxtext = usefultext.split(' [RESPONSE] ')[0]
        context_list = ctxtext.split(' [ENDOFTURN] ')
        # print(ctxtext, '--', response)
        # import pdb;pdb.set_trace()

        dp["dialogue_context"] = context_list
        dp["response_list"] = [response]

    elif dp['task']=='eval_ranking':
        dp['agent_name'] = "test"
        usefultext = dp['input'].split('[CONTEXT] ')[1].split(' [ENDOFDIALOGUE]')[0]
        # response = usefultext.split(' [RESPONSE] ')[1]
        # ctxtext = usefultext.split(' [RESPONSE] ')[0]
        context_list = usefultext.split(' [ENDOFTURN] ')
        dp["dialogue_context"] = context_list
        dp["response_list"] = dp['candidates']
        # import pdb;pdb.set_trace()

    else:
        dp["dialogue_context"] = ['abc']
        dp["response_list"] = ['xyz']


def run_scoring(args):
    server = "http://localhost:" + args.port
    data_json_lines = get_json_lines(args.input_file)
    
    batch_chunks = chunks(data_json_lines, args.batch_size)
    input_file_name = args.input_file.split('/')[-1]

    # os.makedirs(args.output_folder, exist_ok=True)
    count = 0
    with open(args.input_file.replace('.json', '_evalscored'+str(args.port)+'.json'), 'w') as outfile:
        for b, batch in tqdm(enumerate(batch_chunks), total=len(data_json_lines)//args.batch_size):            
            for d,dp in enumerate(batch):
                dp['dialogid'] = str(count)+'_'+str(d)
                get_responsesfromdict(dp) 
            batch_data = batch
            result = requests.post(server,json=batch_data, timeout=1200).text
            # print(batch_data)
            # print(result)
            # import pdb;pdb.set_trace()
            result = json.loads(result)
            # result = json.dumps(json.loads(result),indent=3)
            count+=len(batch)


            # print(outputs)
            for d, dp in enumerate(batch):
                # import pdb;pdb.set_trace()
                batch[d]['metric_score'] = result['Results'][d]['response_scores']
                if b%100==0:
                    print(batch[d]['output'])
                json.dump(batch[d], outfile)
                outfile.write('\n')
                outfile.flush()
            # exit(0)
    # texts =     ["In this course, we will teach you how to", "Is this review positive or negative? Review: Best cast iron skillet you will every buy."]

    # print(generated_texts)




if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)

    run_scoring(args)
