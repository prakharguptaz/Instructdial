import argparse
import json
import logging
import numpy as np
import os
import random
import math
import copy
import time
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict
from typing import Dict, List

from tqdm import tqdm
from tqdm import trange

import settings
from constants import SPECIAL_TOKENS
from datareaders import get_reader
from string import Template, ascii_uppercase
from utils.common import get_options_string, get_alphabetwithoptions_string
from pprint import pprint

from sequentialize import get_sequence
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)



def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--configfile", default='configs/taskfiles_config.json', type=str)
    parser.add_argument("--tasksfiles_folder", default='tasks_files1k/', type=str)    
    parser.add_argument("--outputfile", type=str) #scripts/seq2seqfiles/traindata_tasks1.jsonl
    parser.add_argument("--excluded_task_datasets", default='', type=str)    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_data", type=int, default=-1)
    parser.add_argument("--max_task_size", type=int, default=-1)
    parser.add_argument("--instruction_option_size", type=int, default=400)
    #generally adjust instruction_option_size so the final instruction_option data size equals max_task_size
    parser.add_argument("--instruction_binary_size", type=int, default=1000) 
    #generally keep instruction_binary_size equal to max_task_size
    parser.add_argument('--noshuffle', action='store_true')
    parser.add_argument('--cross_task_options_prob', type=float, default=0, help='The probability of adding options from other tasks')
    parser.add_argument('--none_of_above_prob', type=float, default=0, help='The probability of makeing the output none of above')
    parser.add_argument('--no_instr', action='store_true')
    parser.add_argument('--no_option', action='store_true')

    return parser.parse_args()




def test_readers(args):
    # Data readers
    config = json.load(open(args.configfile, 'r'))
    dataset = args.dataset



def sample_fewshot(data: Dict, k_shot: int, data_dist='uniform') -> List:
    instances = data['Instances']
    samples = []
 
    label2instances = defaultdict(list)
    for instance in instances:
        if 'output' in instance:
            output = instance['output']
        elif 'outputs' in instance:
            output = instance['outputs']
            
        if type(output) == list:
            for label in output:
                label2instances[label].append(instance)
        else:
            label2instances[output].append(instance)
    
    for label, x in label2instances.items():
        samples.append(random.choice(x))
    
    if len(samples) > k_shot:
        samples = samples[:k_shot]
        return samples

    remain_k_shot = k_shot - len(samples)
    print(k_shot, len(samples), remain_k_shot)

    if data_dist == 'uniform':
        data2sample = defaultdict(list)
        for instance in instances:
            data2sample[instance['dataset']].append(instance)
        
        dataset_shot = remain_k_shot // len(data2sample.keys())

        for dataset, dataset_samples in data2sample.items():
            samples += random.sample(dataset_samples, min(dataset_shot, len(dataset_samples)))
        
        remain = k_shot - len(samples)
        if remain > 0:
            samples += random.sample(instances, remain)
    else:
        samples += random.sample(instances, remain_k_shot)
    
    assert len(samples) == k_shot
    return samples


def add_nota_random_option_toinstance(datapoint, task2labels):
    inputtext = datapoint['input']
    try:
        texts = inputtext.split(settings.OPTION_TOKEN)
        context_text = texts[0] 
        if len(texts) == 2:
            option_text = texts[1]
        else:
            option_text = settings.OPTION_TOKEN.join(texts)
    except:
        import ipdb; ipdb.set_trace()
        
    try:
        option_text, question_text = option_text.split(settings.QUESTION_SEP)
    except: # in the case [Question] is before [OPTIONS]
        question_text = None
    cur_options = [x.strip() for x in option_text.split(settings.OPTION_SEPARATOR)]
        

    if settings.OPTION_TOKEN not in inputtext:
        return None
    else:
        # sample optons from other task
        task = datapoint['task']
        is_changed = False
        if random.random() < args.cross_task_options_prob :
            options = []
            for t, labels in task2labels.items():
                if t != task:
                    options.extend(list(labels))
            options = [x for x in labels if x.isalpha()]
            #todo fix it with new option format with integers and alphabets
            extra_options = random.choices(options, k=min(len(options),len(cur_options)))
            extra_options = list(set(extra_options))
            if ':' in cur_options[0] and ':' in cur_options[-1]:
                last_letter = cur_options[-1].split(':')[0]
                if last_letter.isalpha():
                    extra_options = [chr((ord(last_letter.upper())+1+i - 65) % 26 + 65) + ': ' + str(x) for i, x in enumerate(extra_options)]
                else:
                    extra_options = [str(int(last_letter)+1+i) + ': ' + str(x) for i, x in enumerate(extra_options)]
                    # next_letter = str(int(last_letter)+1)
            cur_options.extend(extra_options)
            is_changed = True

        for c in cur_options:
            if '[OPTIONS]' in c:
                print('error')
                import pdb;pdb.set_trace()
        if random.random() < args.none_of_above_prob:
            # try:
            if True:
                if ('A:' not in cur_options[0] and 'B:' not in cur_options[1]) and ('0:' not in cur_options[0] and '1:' not in cur_options[1]):
                    if random.random()<0.5 and ', ' not in datapoint['output']:         #either replace the correct option with nota 
                        cur_options.remove(datapoint['output'].strip())
                        cur_options.append('none of the above')
                        random.shuffle(cur_options)
                        datapoint['output'] = 'none of the above'
                    else:                           #or add nota as extra wothout changing gold so that model does not always predict nota
                        # cur_options.remove(datapoint['output'].strip())
                        candtoremove = random.choice(cur_options)
                        if candtoremove!=datapoint['output'].strip() and ', ' not in candtoremove:
                            if random.random()>0.5:
                                cur_options.remove(candtoremove)
                            else:
                                pass #half of the time we just add and do not remove
                        cur_options.append('none of the above')
                        random.shuffle(cur_options)
                        # datapoint['output'] = 'none of the above'
                    is_changed = True
                else:
                    goldchangeindex = None
                    for c, coption in enumerate(cur_options):
                        if cur_options[c].startswith(str(datapoint['output']).strip()+':'):
                            goldchangeindex = c
                    if random.random()<0.5:           #either replace the correct option with nota
                        cur_options[goldchangeindex] = datapoint['output'].strip()+': none of the above'
                    else:                             #or add nota as extra wothout changing gold so that model does not always predict nota
                        if random.random()>0.5:       # with 50 percent chance add nota
                            last_letter = cur_options[-1].split(':')[0]
                            if last_letter.isalpha():
                                next_letter = chr((ord(last_letter.upper())+1 - 65) % 26 + 65)
                            else:
                                next_letter = str(int(last_letter)+1)
                            cur_options.append(next_letter+': none of the above')
                            indextoconsider = set([i for i in range(len(cur_options))])-set([goldchangeindex])
                            indextochange = random.choice(list(indextoconsider))
                            splited = cur_options[indextochange].split(':')
                            if len(splited)==2:
                                tmpchar, tmpoption = splited
                            else: # typically should not happen
                                tmpchar, tmpoption = splited[0], ':'.join(splited[1:])
                            cur_options[indextochange] = cur_options[indextochange].split(':')[0] + ':' + cur_options[len(cur_options)-1].split(':')[1]
                            cur_options[len(cur_options)-1] = cur_options[len(cur_options)-1].split(':')[0] + ':' + tmpoption
                        else:                       # with 50 percent replace with nota
                            indextoconsider = set([i for i in range(len(cur_options))])-set([goldchangeindex])
                            indextochange = random.choice(list(indextoconsider))
                            cur_options[indextochange] = cur_options[indextochange].split(':')[0] + ': none of the above'
                is_changed = True
            # except Exception as e: # for example, output might contain multiple correct asnwers like "reqmore, recommend"
            #     print('erRor:', e)
            #     import pdb;pdb.set_trace()
            #     pass

        option_text = settings.OPTION_SEPARATOR.join(cur_options)
        nota_text = f". Select none of the above if no valid option is present. "

        if question_text:
            inputtext = f'{context_text} {settings.OPTION_TOKEN} {option_text} {settings.QUESTION_SEP} {question_text} {nota_text} '
        else:
            inputtext = f'{context_text}  {nota_text} {settings.OPTION_TOKEN}  {option_text}'

        datapoint['input'] = inputtext

        if not is_changed: # did not polute the outputs
            return None

    datapoint['task'] = datapoint['task']+'-poluteoptions'

    return datapoint

def encodeinstruction(args, task,
                     data, meta_data_map,
                     few_shot=False, 
                     few_shot_config=None, 
                     number_of_instances=None, 
                     task2labels=None,
                     no_instr=False,
                     no_option=False):

    input_instructionformat=''
    definitions = data['Definitions']

    #gather dataset specific definitions
    dataset_specific_definitions = dict()
    dataset_specific_definition_names = [ k for k in data.keys() if 'Definitions' in k and k!='Definitions' ]
    for dsn in dataset_specific_definition_names:
        dataset_specific_definitions[dsn.split('Definitions-')[-1]] = data[dsn]

    positive_examples = data["Positive Examples"]
    
    if few_shot_config:
        instances = sample_fewshot(data, few_shot_config['k-shot'], few_shot_config['data-dist'])
    else:
        instances = data["Instances"]

    # if few_shot:
    #     for j in range(len(data['Positive Examples'])):
    #         input_instructionformat = input_instructionformat+ '\nInput: '+data['Positive Examples']['Positive Examples'][j]['input'] + ' Output '+data['Positive Examples']['Positive Examples'][j]['output']

    datalist=[]

    if number_of_instances is None:
        number_of_instances = len(instances)

    # print(meta_data_map)
    for i in range(number_of_instances):
        datapoint = instances[i]

        instruction_selected = random.choice(definitions)
        if datapoint['dataset'] in dataset_specific_definitions:
            alt_definitions = dataset_specific_definitions[datapoint['dataset']]
            instruction_selected = random.choice(alt_definitions)

        if no_instr:
            input_instructionformat = f"({task} for {datapoint['dataset']}): "
        else:
            input_instructionformat = "Instruction: "+ instruction_selected + '\n'

        datapoint['task'] = task

        if "output" not in datapoint and "outputs" in datapoint:
            datapoint["output"] = datapoint["outputs"]
            del datapoint["outputs"]
        datapoint["all_outputs"] = copy.deepcopy(datapoint["output"])
        if type(datapoint["output"]) is list:
            datapoint["output"] = random.choice(datapoint["output"])
        if type(datapoint["output"]) is not str:
            datapoint["output"] = str(datapoint["output"])
        if type(datapoint["all_outputs"]) is not list:
            datapoint["all_outputs"] = [str(datapoint["all_outputs"])]        
        if 'outputs' in datapoint:
            del datapoint["outputs"]
        if 'input' not in datapoint:
            datapoint['input'] = datapoint['text']   
        datapoint['text'] = ''
        assert type(datapoint["all_outputs"]) is list
        datapoint["all_outputs"] = [str(x) for x in datapoint["all_outputs"]]

        inputtext = datapoint['input']
        if no_option: # remove options
            if settings.EOD_SEP in inputtext:
                dialog, options = inputtext.split(settings.EOD_SEP)
                inputtext = dialog + settings.EOD_SEP
            elif settings.QUESTION_SEP in inputtext:# only [ENDOFTURN] and [QUESTION]
                inputtext, question = inputtext.split(settings.QUESTION_SEP)
            else: # only [ENDOFTURN]
                dialog = inputtext.split(settings.EOT_SEP)
                inputtext = settings.EOT_SEP.join(dialog[:-1])

        prompt=input_instructionformat+'Input: '+ inputtext #+"\n"+"Output:"

        datapoint['prompt'] = prompt

        for keyname in ['classes_in_options']: 
            if keyname not in datapoint:
                datapoint[keyname] = []
        for keyname in ['candidates']:
            if keyname not in datapoint:
                datapoint[keyname] = []
        for keyname in ['metadata']:
            if keyname not in datapoint:
                datapoint[keyname] = {}
        if 'metadata' not in datapoint:
            datapoint['metadata'] = {}
        if 'context' in datapoint['metadata'] and type(datapoint['metadata']['context']) is not list:
            datapoint['metadata']['context'] = [datapoint['metadata']['context']]

        # add necessary key values in metadata fields to satisfy pyarrow
        for k in meta_data_map.keys():
            if k not in datapoint['metadata'] and type(meta_data_map[k]) is not dict:
                datapoint['metadata'][k] = None
            if type(meta_data_map[k]) is dict:
                if k not in datapoint['metadata']:
                    datapoint['metadata'][k] = dict()
                for ink in meta_data_map[k].keys():
                    if k not in datapoint['metadata']:
                        datapoint['metadata'][k] = {}
                    if ink not in datapoint['metadata'][k]:
                        if k =='human_rating':
                            datapoint['metadata'][k][ink] = -1000.0
                        else:
                            datapoint['metadata'][k][ink] = None
        # datapoint['metadata'] = {'context': datapoint['metadata'].get('context', []), 'keywords': datapoint['metadata'].get('keywords', []), 'knowledge': datapoint['metadata'].get('knowledge', None), 'beginswith': datapoint['metadata'].get('beginswith', None), 'persona': datapoint['metadata'].get('persona', None), 'endswith': datapoint['metadata'].get('endswith', None), 'target': datapoint['metadata'].get('target', None), 'domain': datapoint['metadata'].get('domain', ''), 'slot_label': datapoint['metadata'].get('slot_label', ''), 'response': datapoint['metadata'].get('response', ''), 'human_rating':datapoint['metadata'].get('human_rating', None), 'eval_dataset_name':datapoint['metadata'].get('eval_dataset_name', '')}
        # datapoint['text'] = datapoint['metadata'].get('text', '')
        # add new options

        if task2labels and datapoint['task'] not in ['eval_binary', 'eval_ranking'] and args.no_instr is False:
            #current options
            dpdatapoint = copy.deepcopy(datapoint)
            dpdatapoint = add_nota_random_option_toinstance(dpdatapoint, task2labels)
            if dpdatapoint is not None:
                dpdatapoint['prompt'] = input_instructionformat+'Input: '+ dpdatapoint['input']
                datalist.append(dpdatapoint)


        datalist.append(datapoint)


    return datalist, definitions


def generate_instruction_option_taskdata(instruction_option_sampledata, all_definitions, max_definition_options = 3, no_instr=False, no_option=False):
    '''
    Generate dataset for a task where model has to choose and instruction given the input and output
    '''

    #no need of generating this data if only one task is being used for training
    if len(all_definitions)<2:
        return []
    list_all_definitions = set().union(*all_definitions.values())
    print('number of definitions present for instruction option task', len(list_all_definitions))
    new_instruction_task_data = []
    for i, olddp in enumerate(instruction_option_sampledata):
        dp = copy.deepcopy(olddp)
        dp_task = dp['task']
        if 'poluteoptions' in dp_task: continue
        task_definitions = all_definitions[dp_task]
        incorrect_definitions = list_all_definitions - set(task_definitions)
        incorrect_definitions_list = list(incorrect_definitions)
        chosen_incorrect_definitions = random.sample(incorrect_definitions_list, min(len(incorrect_definitions_list), max_definition_options))
        chosen_correct_definition = random.choice(task_definitions)

        all_indices = range(len(chosen_incorrect_definitions)+1)
        answer_idx = random.choice(all_indices)
        candidates = chosen_incorrect_definitions[:answer_idx] + [chosen_correct_definition] + chosen_incorrect_definitions[answer_idx:]
        assert candidates[answer_idx] == chosen_correct_definition
        answer_str = get_alphabetwithoptions_string(candidates)
        candidate_options = []
        for option, candidate in zip(ascii_uppercase, candidates):
            candidate_options.append(f'{option}')

        if type(dp['all_outputs']) is not list:
            import pdb;pdb.set_trace()
        original_output = random.choice(dp['all_outputs'])

        output = ascii_uppercase[answer_idx]
        dp['all_outputs'] = [output]
        dp['output'] = output
        dp['task'] = 'instruction_selection'

        if 'input' in dp:
            original_text = dp['input']
        else:
            original_text = dp['text']
        original_text = original_text.split(settings.QUESTION_SEP)[0].split(settings.EOD_SEP)[0]

        task_definitions = ['Instruction: In this task you will be shown a dialogue input and a output of a task applied on the dialogue input. You need to select the most appropiate instruction for the task given the dialogue, the output and the instruction options\nInput: ', 'Instruction: In this task given a dialogue input and the output corresponding to a task for the dialogue, choose the best instruction for the task among the provided options\nInput: '] 

        dp['prompt'] = random.choice(task_definitions) + original_text + '. The output is: ' + settings.OUTPUT_SEP+' '+ str(original_output) + ' ' +settings.QUESTION_SEP +random.choice(['. The instructions options are: ','. The list of options for instructions are: ', '. The possible instructions are: ']) + answer_str + ' [QUESSTION] ' + random.choice(['. The correct option among instructions is:', '. The instruction which can lead to the output given the dialogue is:', '. The correct choice of instruction is: '])
        dp['classes_in_options'] = candidate_options
        dp['candidates'] = candidates

        new_instruction_task_data.append(dp)
        # print(dp)
        # import pdb;pdb.set_trace() 

    return new_instruction_task_data


def print_distribution(all_data):
    tasksizedict = defaultdict(int)
    for dp in all_data:
        tasksizedict[dp['task']]+=1

    print(tasksizedict)

    return tasksizedict

def generate_instruction_binary_taskdata(instruction_option_sampledata, all_definitions, max_definition_options = 3, no_instr=False, no_option=False):
    '''
    Generate dataset for a task where model has to predict if givent he input and output, the provided instruction is valid
    '''

    #no need of generating this data if only one task is being used for training
    if len(all_definitions)<2:
        return []
    list_all_definitions = set().union(*all_definitions.values())
    # print('number of definitions present for instruction binary task', len(list_all_definitions))
    new_instruction_task_data = []
    for i, olddp in enumerate(instruction_option_sampledata):
        dp = copy.deepcopy(olddp)
        dp_task = dp['task']
        if 'poluteoptions' in dp_task: continue
        task_definitions = all_definitions[dp_task]
        incorrect_definitions = list_all_definitions - set(task_definitions)
        incorrect_definitions_list = list(incorrect_definitions)
        chosen_incorrect_definitions = random.sample(incorrect_definitions_list, min(len(incorrect_definitions_list), max_definition_options))
        chosen_correct_definition = random.choice(task_definitions)

        # all_indices = range(len(chosen_incorrect_definitions))
        # answer_idx = random.choice(all_indices)
        # candidates = chosen_incorrect_definitions[:answer_idx] + [chosen_correct_definition] + chosen_incorrect_definitions[answer_idx:]
        # assert candidates[answer_idx] == chosen_correct_definition
        candidates = ['yes', 'no']        
        answer_str = get_alphabetwithoptions_string(candidates)
        candidate_options = []
        for option, candidate in zip(ascii_uppercase, candidates):
            candidate_options.append(f'{option}')

        original_output = random.choice(dp['all_outputs'])


        if 'input' in dp:
            original_text = dp['input']
        else:
            original_text = dp['text']
        original_text = original_text.split(settings.QUESTION_SEP)[0].split(settings.EOD_SEP)[0]

        task_definitions = ['Instruction: You will be shown an input and an output of a task along with the insruction for the task. You need to choose if the instruction fits the task given the dialogue input and the output\nInput: ', 'Instruction: In this task given a dialogue input and the output corresponding to a task for the dialogue, choose if the given instruction is appropriate for the task options\nInput: ']
        dp['classes_in_options'] = candidate_options
        dp['candidates'] = candidates
        dp['task'] = 'instruction_binary'

        if random.random()<0.5:
            if no_instr:
                instruction = ''
            else:
                instruction = ' [INSTRUCTION] ' + chosen_correct_definition
            output = 'yes'
            dp['prompt'] = random.choice(task_definitions) + instruction + original_text + '. The output is: ' + settings.OUTPUT_SEP+' '+ original_output + ' ' +random.choice(['. The instructions options are: ','. The list of options are: ', '. The possible options are: ']) + answer_str + f' {settings.QUESTION_SEP} ' + random.choice(['. Does the instruction fits the task?', '. The instruction can lead to the output given the dialogue input', '. The instruction is correct for the input and output '])
            dp['all_outputs'] = [output]
            dp['output'] = output
        else:
            instruction = ' [INSTRUCTION] ' + random.choice(chosen_incorrect_definitions)
            output = 'no'
            dp['prompt'] = random.choice(task_definitions) + instruction + original_text + '. The output is: ' + settings.OUTPUT_SEP+' '+ original_output + ' ' +settings.QUESTION_SEP +random.choice(['. The instructions options are: ','. The list of options are: ', '. The possible options are: ']) + answer_str + f' {settings.QUESTION_SEP} ' + random.choice(['. Does the instruction fits the task?', '. The instruction can lead to the output given the dialogue input', '. The instruction is correct for the input and output '])
            dp['all_outputs'] = [output]
            dp['output'] = output

        new_instruction_task_data.append(dp)

    return new_instruction_task_data

def get_task_outputs(args, task, data):
    
    instances = data["Instances"]
    
    task2labels = defaultdict(set)
    for instance in instances:
        if 'input' in instance:
            inputtext = instance['input']
        else:
            inputtext = instance['text']


        if 'output' in instance:
            output = instance['output']
        elif 'outputs' in instance:
            output = instance['outputs']
            
        if type(output) == list:
            for label in output:
                output = label


        if settings.OPTION_TOKEN not in str(inputtext) and len(str(inputtext))>20:
            continue
                # task2labels[task].add(label)
        # else:
        #     task2labels[task].add(output)
        task2labels[task].add(output)
    return task2labels
    
def encode_tasks(args):
    config = json.load(open(args.configfile, 'r'))
    taskfiles_list = config['task-files']

    global_datasets_excluded = config.get('datasets_excluded', [])
    #task_datasets_details contains options list of datasets to include or exclude per task 
    task_datasets_details = config.get('task_datasets_details', {})

    all_data = []
    all_definitions = {}
    instruction_option_sampledata = []
    instruction_binary_sampledata = []
    
    with open('task-data-configs.txt', 'a') as f:
        json.dump(config, f)
        f.write('\nout:'+args.outputfile+'\n')

    few_shot_tasks = config.get('few_shot_tasks', None)

    # get all instruction
    task2labels = {}
    task2data = {}
    meta_data_map = {}
    all_task_list = taskfiles_list
    if few_shot_tasks is not None:
        all_task_list+=list(few_shot_tasks.keys())
    for task in all_task_list:
        with open(args.tasksfiles_folder+task+'.json') as json_file:
            data = json.load(json_file)
            # data['Instances'] = data['Instances'][:3]

        #remove or add datasets set in task_datasets_details
        if len(global_datasets_excluded)>0 or (len(task_datasets_details)>0 and task in task_datasets_details):
            task_dataset_config = task_datasets_details.get(task, {})
            filtered_data = []
            #datasets_to_include and datasets_to_exclude are optional keys
            datasets_to_include = task_dataset_config.get('datasets-included', [])
            datasets_to_exlude = task_dataset_config.get('datasets-excluded', []) + global_datasets_excluded
            print(f"For {task}, datasets_to_include {datasets_to_include} and datasets_to_exlude {datasets_to_exlude}")
            prev_task_size = len(data['Instances'])
            if len(datasets_to_include)>0:
                data['Instances'] = [x for x in data['Instances'] if x['dataset'] in datasets_to_include]
            #datasets_to_exclude will remove the instances even if they are present in datasets_to_include
            data['Instances'] = [x for x in data['Instances'] if x['dataset'] not in datasets_to_exlude]
            if prev_task_size!=len(data['Instances']):
                print(f"---Size was " + str(prev_task_size) + f" for task {task}, before running task_dataset_config")
                print(f'---Size now ' + str(len(data['Instances'])))

        #update task2labels
        task2data[task] = data
        task2labels.update(get_task_outputs(args, task, data))

        # create a common metadata dictionary that should be present in all tasks data
        max_text_size = -1
        for i in range(len(data['Instances'])):
            datapoint = data['Instances'][i]
            metadata_dp = datapoint.get('metadata')
            if 'input' not in datapoint:
                datapoint['input'] = datapoint['text'] 
            max_text_size = max(max_text_size, len(datapoint['input'].split()))
            if metadata_dp is not None:
                for k in metadata_dp.keys():
                    value = metadata_dp[k]
                    if type(value) is dict:
                        if k not in meta_data_map:
                            meta_data_map[k] = dict()
                        for ink, inv in value.items():
                            meta_data_map[k][ink] = type(inv)
                    if k not in meta_data_map:
                        meta_data_map[k] = type(value)
        print(task, 'max_input_len', max_text_size)
    
    for task in taskfiles_list:
        if few_shot_tasks and task in few_shot_tasks:
            continue
        data = task2data[task]
        datalist, definitions = encodeinstruction(args, task, data, meta_data_map, task2labels=task2labels, no_instr=args.no_instr, no_option=args.no_option)
        all_definitions[task] = definitions
        if args.max_task_size!=-1:
            if task not in ['gensf_slot_tagging']:
                datalist = random.sample(datalist, min(len(datalist),args.max_task_size))

        #select data for instruction_option task
        if args.instruction_option_size!=-1:
            instruction_option_sampledata+= random.sample(datalist, min(len(datalist),args.instruction_option_size))
        #select data for instruction_binary task
        if args.instruction_binary_size!=-1:
            instruction_binary_sampledata+= random.sample(datalist, min(len(datalist),args.instruction_binary_size))
            
        all_data+=datalist
        print(task, " with number of datapoints:", len(datalist))

    if few_shot_tasks:
        for task, few_shot_config in few_shot_tasks.items():
            data = task2data[task]
            datalist, definitions = encodeinstruction(args, task, data, meta_data_map, few_shot_config=few_shot_config, no_instr=args.no_instr, no_option=args.no_option)
            all_definitions[task] = definitions
            if args.max_task_size!=-1:
                datalist = random.sample(datalist, min(len(datalist),args.max_task_size))

            all_data+=datalist
            print(task, " with number of fewshot datapoints:", len(datalist))

    if args.noshuffle is False:
        random.shuffle(all_data)

    print('Length of all data:', len(all_data))

    if args.max_data!=-1:
        all_data = random.sample(all_data, min(len(all_data),args.max_data))
        print('Sampled Length of all data:', len(all_data))

    if args.instruction_option_size>-1 and args.no_instr is False:
        instruction_option_sampledata = random.sample(instruction_option_sampledata, min(len(instruction_option_sampledata),args.instruction_option_size))
        new_instruction_task_data = generate_instruction_option_taskdata(instruction_option_sampledata, all_definitions, no_instr=args.no_instr, no_option=args.no_option)
        all_data+=new_instruction_task_data
        print('instruction option task data len', len(new_instruction_task_data))
    if args.instruction_binary_size>-1 and args.no_instr is False:
        instruction_binary_sampledata = random.sample(instruction_binary_sampledata, min(len(instruction_binary_sampledata),args.instruction_binary_size))
        new_instruction_task_data = generate_instruction_binary_taskdata(instruction_binary_sampledata, all_definitions, no_instr=args.no_instr, no_option=args.no_option)
        all_data+=new_instruction_task_data
        print('instruction binary task data len', len(new_instruction_task_data))

    task_disstribution_dict = print_distribution(all_data)
    print('all_data', len(all_data))
    with open('task-data-configs.txt', 'a') as f:
        json.dump(task_disstribution_dict, f)
        f.write('\n\n')

    dest_file = args.outputfile
    output_file = open(dest_file, 'w', encoding='utf-8')
    for dic in all_data:
        json.dump(dic, output_file) 
        output_file.write("\n")

    return encode_tasks


if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)
    startTime = time.time()

    encode_tasks(args)
    
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

#python -m scripts.create_data_seq2seq --outputfile scripts/seq2seqfiles/traindata_tasks1.json
#python -m scripts.create_data_text2text --outputfile scripts/text2textfiles/traindata_debug1.json --configfile configs/debug.json --tasksfiles_folder tasks_files/tasks_files-full-trainconfig1/  --max_task_size 1100 --cross_task_options_prob 0.0 --none_of_above_prob 0.0 --instruction_option_size 50
