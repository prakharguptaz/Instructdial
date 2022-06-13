import sys
import json
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
    data_json_lines = get_json_lines(args.input_file)
    
    batch_chunks = chunks(data_json_lines, args.batch_size)
    input_file_name = args.input_file.split('/')[-1]

    # os.makedirs(args.output_folder, exist_ok=True)
    count = 0
    all_metric_corrects_eval_ranking = defaultdict(list)
    all_metric_preds_eval_ranking = defaultdict(list)
    all_metric_scores_eval_binary_yes = defaultdict(list)
    all_metric_scores_eval_binary_no = defaultdict(list)

    all_metric_scores_eval_binary_dict = defaultdict(dict)

    for d,dp in enumerate(data_json_lines):
        label = dp['all_outputs'][0]
        responses = dp['response_list']
        if dp['task']=='eval_rating':
            continue
        elif dp['task']=='eval_ranking':
            metric_score_names = list(dp['metric_score'][0].keys())
            for m in metric_score_names:
                mscores = [dp['metric_score'][rid][m] for rid, r in enumerate(responses)]
                max_value = max(mscores)
                max_index = mscores.index(max_value)
                pred_letter = chr(ord('@')+(max_index+1))
                iscorrect = label==pred_letter
                all_metric_corrects_eval_ranking[m].append(iscorrect)
                all_metric_preds_eval_ranking[m].append(pred_letter)
        elif dp['task']=='eval_binary':
            metric_score_names = list(dp['metric_score'][0].keys())
            bkey = dp['task'] + '--' + '--' + '++'.join(dp['dialogue_context']) 
            for m in metric_score_names:
                mscore = [dp['metric_score'][rid][m] for rid, r in enumerate(responses)][0]
                all_metric_scores_eval_binary_dict[bkey][label] = mscore
                if label == 'yes':
                    all_metric_scores_eval_binary_yes[m].append(mscore)
                if label== 'no':
                    all_metric_scores_eval_binary_no[m].append(mscore)


    for m in all_metric_corrects_eval_ranking:
        print(m, sum(all_metric_corrects_eval_ranking[m])/len(all_metric_corrects_eval_ranking[m]), len(all_metric_corrects_eval_ranking[m]))

    corrects_list = []
    for bkey in all_metric_scores_eval_binary_dict:
        dpbinary_dict = all_metric_scores_eval_binary_dict[bkey]
        # print(dpbinary_dict)        
        if len(dpbinary_dict.keys())==2:
            # gold_score = dpbinary_dict['yes'] / (dpbinary_dict['yes']+dpbinary_dict['no'] + 1e-12)
            # print(dpbinary_dict, gold_score)
            # if gold_score>0.5:
            #     corrects_list.append(1)
            # else:
            #     corrects_list.append(0)
            # import pdb;pdb.set_trace()
            if dpbinary_dict['yes']>dpbinary_dict['no']:
                corrects_list.append(1)
            else:
                corrects_list.append(0)

    if len(corrects_list)>0:
        print('Eval binary report: percentage correct:', sum(corrects_list)/len(corrects_list), 'len', len(corrects_list))

    print('\nall_metric_scores_eval_binary')
    for m in all_metric_scores_eval_binary_yes:
        print(m, 'yes',sum(all_metric_scores_eval_binary_yes[m])/len(all_metric_scores_eval_binary_yes[m]), len(all_metric_scores_eval_binary_yes[m]))
        print(m, 'no',sum(all_metric_scores_eval_binary_no[m])/len(all_metric_scores_eval_binary_no[m]), len(all_metric_scores_eval_binary_no[m]))


                # for rid, r in enumerate(responses):
                #     metric_score_dict = dp['metric_score'][rid]
                




if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)

    run_scoring(args)
