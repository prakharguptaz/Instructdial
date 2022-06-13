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
import jsonlines
from tqdm import tqdm
from tqdm import trange

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from eval_scripts.classification_metrics import correlation_metrics
from eval_scripts.classification_metrics import accuracy_metric, evaluate_slotrestaurant8k
from eval_scripts.generation_metrics import begins_with_metric, ends_with_metric, response_length_metric
from eval_scripts.generation_metrics import nlgeval_metrics, bert_score, bleurt, usl_score, calc_hfbleu_scores, calc_rouge_scores
from eval_scripts.task_oriented_metrics import  joint_accuracy

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)



nlg_ref_metrics = [
    nlgeval_metrics, calc_hfbleu_scores, calc_rouge_scores,#, bert_score, 
]
#metrics which can be added - bleurt

nlg_reffree_metrics = [
    usl_score
]

classification_metrics_list = [classification_report, accuracy_score, accuracy_metric, precision_score, recall_score, f1_score]

task_metrics_map = {
    'intent_classification': classification_metrics_list,
    'intent_classification_banking': classification_metrics_list,
    'relation_classification': classification_metrics_list,
    'dialfact_classification': classification_metrics_list,
    'intent_present': classification_metrics_list,
    'answer_selection': classification_metrics_list,
    'eval_rating': classification_metrics_list,
    'relation_present': classification_metrics_list,
    'emotion_tagging': classification_metrics_list,
    'eval_binary': [correlation_metrics],#+classification_metrics_list,
    'eval_ranking': classification_metrics_list,
    'nli_classification': classification_metrics_list,
    'act_classification': classification_metrics_list,
    'persuasion_present': classification_metrics_list,
    'advice_present': classification_metrics_list,
    'deal_present': classification_metrics_list,
    'slot_present': classification_metrics_list,
    'slot_value_generation': classification_metrics_list,
    'slot_value_generation_rst8k': [evaluate_slotrestaurant8k]+classification_metrics_list,
    'slot_tagging': classification_metrics_list,
    'gensf_slot_tagging': [evaluate_slotrestaurant8k]+classification_metrics_list, 
    'persuasion_strategy': classification_metrics_list,
    'keyword_controlled_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'document_grounded_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'beginswith_controlled_generation': [nlgeval_metrics, begins_with_metric] + nlg_ref_metrics,
    'endswith_controlled_generation': [nlgeval_metrics, ends_with_metric] + nlg_ref_metrics,
    'target_controlled_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'answer_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'advice_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'question_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'persuasion_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'schema_based_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'db_based_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'graph_based_generation': [nlgeval_metrics],
    'emotion_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'act_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'knowledge_grounded_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'response_generation_with_n_words': [nlgeval_metrics] + nlg_ref_metrics + [response_length_metric],
    'persona_grounded_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'response_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'belief_state_generation': [joint_accuracy] + nlg_ref_metrics,
    'summarization': [nlgeval_metrics] + nlg_ref_metrics,
    'nontoxic_feedback_generation': [nlgeval_metrics] + nlg_ref_metrics,
    'fill_missing_utterance': [nlgeval_metrics] + nlg_ref_metrics,

}


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--outputfile", type=str) #scripts/seq2seqfiles/traindata_tasks1.jsonl
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_mode", type=str, default='all') #all,per_dataset    
    parser.add_argument('--instructionmetrics', action='store_true')

    return parser.parse_args()



def get_json_lines(inp_file):
    lines = []
    with jsonlines.open(inp_file) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

def test_readers(args):
    # Data readers
    config = json.load(open(args.configfile, 'r'))
    dataset = args.dataset

def backupslotfinder(text):
    #["time", "people", "first_name", "last_name", "date"]
    if 'slot time' in text or 'slot: time' in text or 'of time' in text:
        return 'time'
    elif 'slot date' in text or 'slot: date' in text or 'of date' in text:
        return 'date'
    elif 'slot people' in text or 'slot: people' in text or 'of people' in text:
        return 'people'
    elif 'first_name' in text or 'first name' in text:
        return 'first-name'
    elif 'last_name' in text or 'last name' in text:
        return 'last-name'
    else:
        import pdb;pdb.set_trace()

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_responseforslot(dp):
    if dp['metadata'].get('response', '')=='':
        markerrindex = dp['input'].rindex('[ENDOFDIALOGUE]')
        responserindex = dp['input'].rindex('[RESPONSE]')
        inputtext = dp['input'][responserindex+1:markerrindex]
    else:
        inputtext = dp['metadata'].get('response', '')

    return inputtext

def closest(pred, text, label):
    # try all subsets of the text

    words = text.split()
    best, best_score = "", 100
    for start in range(len(words)):
        for end in range(start+1, len(words)+1):
            subset = " ".join(words[start:end])
            score = levenshteinDistance(pred, subset)
            if score < best_score:
                best, best_score = subset, score

    # if ('pm' in text.lower() or 'am' in text.lower() or '.' in pred.lower()) and pred!='not present':
    #     print(best, 'll', label)
    #     import pdb;pdb.set_trace()

    if best_score < 0.3*len(pred):
        return best
    else:
        return ""

def get_humanrating(datapoint):
    dp_ratings = datapoint['metadata']['human_rating']
    rating = None
    if type(dp_ratings) is dict and len(dp_ratings.keys())>0:
        if 'overall' in dp_ratings  and dp_ratings['overall'] is not None:
            rating = dp_ratings['overall']
        elif 'turing' in dp_ratings  and dp_ratings['turing'] is not None:
            rating = dp_ratings['turing']
        elif 'relevance' in dp_ratings  and dp_ratings['relevance'] is not None:
            rating = dp_ratings['relevance']
        elif 'appropriateness' in dp_ratings  and dp_ratings['appropriateness'] is not None:
            rating = dp_ratings['appropriateness']
    if rating is None:
        import pdb;pdb.set_trace()

    return rating


def clean_prediction(prediction):
    prediction = prediction.lower()
    if prediction == 'yesyes':
        prediction = 'yes'
    if prediction == 'nono':
        prediction = 'no'

    return  prediction

def calculate_task_metrics(task, task_data, metrics_dict, dataset_level=False, instruction_level=False):
    if task not in task_metrics_map:
        print(task, ' - metrics not defined')
        return None

    # print('Metrics', task_metrics_map[task])
    for metric in task_metrics_map[task]:
        metricname = metric.__name__
        # print('METRIC:', metricname)
        if metric in [classification_report, accuracy_score, accuracy_metric, precision_score, recall_score, f1_score]:
            y_true, y_pred = [], []
            y_true_classnames, y_pred_classnames = [], []
            for dp in task_data:
                prediction = str(dp['output'])
                possible_classes = dp.get('classes_in_options', [])
                label = dp['all_outputs']
                if type(label) is list and len(label)==1:
                    label = label[0]
                # we do not support multi-abe; classification, and if a hypothesis matches any label, we mark it as correct
                if type(label) is list and len(label)>1:
                    for l in label:
                        if prediction in l:
                            label = l
                    if type(label) is list:
                        label = random.choice(label)

                label = str(label).lower()
                if task=='eval_binary' and label=='ambiguous':
                    continue
                # if label=='not present': continue
                y_true.append(label)
                final_pred = clean_prediction(prediction)
                y_pred.append(final_pred)
                dp['final_pred'] = final_pred
                dp['final_label'] = label

                #add classnames for some tasks
                # if task in ['dialfact_classification'] and 'candidates' in dp:
                #     candidates_names = [x.lower() for x in dp['candidates']]
                #     classes_in_options_names = [x.lower() for x in dp['classes_in_options']]
                #     label_index = classes_in_options_names.index(label)
                #     label_classname = candidates_names[label_index]
                #     pred_index = classes_in_options_names.index(final_pred)
                #     pred_classname = candidates_names[pred_index]
                #     y_true_classnames.append(label_classname)
                #     y_pred_classnames.append(pred_classname)

            # if one needs to get class_label based classification reports
            # if len(y_pred_classnames)>0:
            #     y_pred = y_pred_classnames[:]
            #     y_true = y_true_classnames[:]

            out_of_label_pred = 0
            if dataset_level ==False:
                for dp in task_data:
                    if dp['final_pred'] not in y_true:
                        out_of_label_pred+=1
                    metrics_dict[task]['out_of_label_pred'] = out_of_label_pred/len(y_pred)
            if metric in [precision_score, recall_score, f1_score]:
                score = metric(y_true, y_pred, average='weighted',zero_division=0)
                metrics_dict[task][metricname+'_weighted'] = score
                score = metric(y_true, y_pred, average='micro',zero_division=0)
                metrics_dict[task][metricname+'_micro'] = score
            elif metric in [classification_report]:
                target_names = list(set(y_true))
                cs_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                metrics_dict[task]['classification_report'] = cs_report
            else:
                score = metric(y_true, y_pred)
                metrics_dict[task][metricname] = score
        elif metric in [correlation_metrics]:
            y_true, y_pred = [], []
            if 'class_probability' not in task_data[0]:
                print('class probability not found for correlation')
                continue
            for dp in task_data:
                prediction = str(dp['output'])
                possible_classes = dp.get('classes_in_options', [])
                label = dp['all_outputs']
                if type(label) is list and len(label)==1:
                    label = label[0]
                label = str(label).lower()
                if task=='eval_binary':
                    label_index = possible_classes.index('yes')
                    # if dp['metadata']['eval_dataset_name']!='dstc7':
                    #     continue
                # if label=='not present': continue
                    yes_prob = dp['class_probability'][label_index]
                    human_rating = get_humanrating(dp)
                    if human_rating is not None:
                        y_true.append(human_rating)
                        y_pred.append(yes_prob)
            crmetrics = correlation_metrics(y_true, y_pred)
            print(crmetrics, len(y_true))
            metrics_dict[task]['spearmanr_res'] = crmetrics[0]
            metrics_dict[task]['pearsonr_res'] = crmetrics[1]

        elif metric in [evaluate_slotrestaurant8k]:
            y_true, y_pred = [], []
            preds, trues = [], []
            dfd_preds, dfd_labels = defaultdict(dict), defaultdict(dict)
            all_slot_names = set()
            for dp in tqdm(task_data[:]):
                # import pdb;pdb.set_trace()
                prediction = str(dp['output'])
                possible_classes = dp.get('classes_in_options', [])
                label = dp['all_outputs']
                if type(label) is list and len(label)==1:
                    label = label[0]
                inputtext = get_responseforslot(dp)
                slot_given = dp['metadata'].get('slot_label', '')

                # if dp['metadata']['domain']!='' and dp['metadata']['domain']!='RentalCars_1':
                #     continue
                # print(prediction, '---', label)
                if 'rst' in task:
                    predictedclose = closest(prediction.strip(), inputtext, label)
                    # if len(prediction)>1 and prediction!='not present':
                    #     print(prediction, '-', predictedclose, '--', label, '---', inputtext)
                    predicted = predictedclose
                else:
                    predicted = prediction.strip()
                    # if len(prediction)>1 and prediction!='not present':
                    #     print(prediction, '--', label, '---', inputtext)
                    if len(prediction)>1 and label!=prediction and 'not' not in label:#'not present'# and 'date' in slot_given:
                        print(prediction, '--', label,slot_given, '---', inputtext)
                label = str(label).lower()
                predicted = str(predicted).lower()
                if 'not' in predicted: predicted = ''
                if 'not' in label: label = ''
                # print(slot_given)
                # import pdb;pdb.set_trace()
                if slot_given == '':
                    slot_given = backupslotfinder(dp['input'][dp['input'].rindex('[QUESTION]'):])
                # if label=='not present': continue
                all_slot_names.add(slot_given)

                y_true.append(label)
                y_pred.append(predicted) 
                pred_dict,label_dict = {slot_given:predicted}, {slot_given:label}
                if predicted=='':
                    pred_dict = {}
                if label=='':
                    label_dict = {}
                dfd_preds[dp['index']][slot_given] = predicted
                dfd_labels[dp['index']][slot_given] = label
                preds.append(pred_dict)
                trues.append(label_dict)

            trues_list = []
            preds_list = []

            for k,v in dfd_preds.items():
                v = {sk: sv for sk, sv in v.items() if sv!=''}
                preds_list.append(v)
                tmplabels = dfd_labels[k]
                tmplabels = {sk: sv for sk, sv in tmplabels.items() if sv!=''}
                trues_list.append(tmplabels)
            print('all_slot_names', all_slot_names)
            # import pdb;pdb.set_trace()
            mscores = metric(preds_list, trues_list, slot_types=all_slot_names)
            print(mscores)
            metrics_dict[task][metricname] = mscores     

        elif metric in nlg_ref_metrics:
            y_true, y_pred = [], []
            for dp in task_data:
                y_pred.append(dp['output'])
                y_true.append(dp['all_outputs'])
            # import pdb;pdb.set_trace()
            if metric in [nlgeval_metrics]:
                score_dict = metric(y_true, y_pred)
                for metric_key in score_dict:
                    metric_list_score = score_dict[metric_key]
                    score = sum(metric_list_score)/len(metric_list_score)
                    metrics_dict[task][metric_key] = score
            else:# other metrics in nlg_ref_metrics:
                score_dict = metric(y_true, y_pred)
                for metric_key in score_dict:
                    metric_list_score = score_dict[metric_key]
                    if type(metric_list_score) is not list:
                        metrics_dict[task][metric_key] = metric_list_score
                    else:  
                        score = sum(metric_list_score)/len(metric_list_score)
                        metrics_dict[task][metric_key] = score
        elif metric in nlg_reffree_metrics:
            pass

        elif metric in [begins_with_metric]:
            y_true, y_pred = [], []
            phrases = []
            for dp in task_data:
                prediction = str(dp['output']).lower()
                y_pred.append(prediction)
                possible_classes = dp.get('classes_in_options', [])
                label = dp['all_outputs']
                if type(label) is list and len(label)==1:
                    label = label[0]
                label = str(label).lower()
                y_true.append(label)
                phrase = dp['metadata']['beginswith']
                phrases.append(phrase)
            begins_score = begins_with_metric(y_pred, y_true, phrases)
            metrics_dict[task][metricname] = begins_score
        elif metric in [ends_with_metric]:
            y_true, y_pred = [], []
            phrases = []
            for dp in task_data:
                prediction = str(dp['output']).lower()
                y_pred.append(prediction.strip())
                possible_classes = dp.get('classes_in_options', [])
                label = dp['all_outputs']
                if type(label) is list and len(label)==1:
                    label = label[0]
                label = str(label).lower()
                y_true.append(label)
                phrase = dp['metadata']['endswith'].strip()
                phrases.append(phrase)
            begins_score = ends_with_metric(y_pred, y_true, phrases)
            metrics_dict[task][metricname] = begins_score
        elif metric in [response_length_metric]:
            y_true, y_pred = [], []
            for dp in task_data:
                prediction = str(dp['output']).lower()
                y_pred.append(prediction.strip())
                possible_classes = dp.get('classes_in_options', [])
                label = dp['all_outputs']
                if type(label) is list and len(label)==1:
                    label = label[0]
                label = str(label).lower()
                y_true.append(label)
            begins_score = response_length_metric(y_pred, y_true)
            metrics_dict[task][metricname] = begins_score
        elif metric in [joint_accuracy]:
            joint_acc = joint_accuracy(task_data)
            metrics_dict[task][metricname] = joint_acc



def calculate_metrics(args):
    test_outputs = get_json_lines(args.outputfile)

    task_data_dict = defaultdict(list)
    for i, dp in enumerate(test_outputs):
        task_data_dict[dp['task']].append(dp)

    print('Tasks found', list(task_data_dict.keys()))

    metrics_dict = defaultdict(dict)

    for task in task_data_dict.keys():
        print('Task', task, 'number of datapoints', len(task_data_dict[task]))
        metrics_dict[task]['num_datapoinst'] = len(task_data_dict[task])
        calculate_task_metrics(task, task_data_dict[task], metrics_dict)

        #to get dataset level metrics
        task_datasetspecific_dict = defaultdict(list)
        for i, dp in enumerate(task_data_dict[task]):
            task_datasetspecific_dict[dp['dataset']].append(dp)
            if 'metadata' in dp and 'eval_dataset_name' in dp['metadata'] and dp['metadata']['eval_dataset_name'] is not None:
                task_datasetspecific_dict[dp['metadata']['eval_dataset_name']].append(dp)
        # print(task, task_datasetspecific_dict.keys())
        metrics_dict[task]['dataset_metrics'] = dict()  
        for dataset_name in task_datasetspecific_dict.keys():
            metrics_dict[task]['dataset_metrics'][dataset_name] = defaultdict(dict)
            metrics_dict[task]['dataset_metrics'][dataset_name]['num_datapoints'] = len(task_datasetspecific_dict[dataset_name])
            calculate_task_metrics(task, task_datasetspecific_dict[dataset_name], metrics_dict[task]['dataset_metrics'][dataset_name],dataset_level=True)

        #to get instruction specific metrics
        if args.instructionmetrics:
            task_instruction_specific_dict = defaultdict(list)
            for i, dp in enumerate(task_data_dict[task]):
                instructiontext = dp['prompt'].split('\nInput:')[0]
                task_instruction_specific_dict[instructiontext].append(dp)
            # print(task, task_datasetspecific_dict.keys())
            metrics_dict[task]['instruction_metrics'] = dict()  
            for inst_name in task_instruction_specific_dict.keys():   
                metrics_dict[task]['instruction_metrics'][inst_name] = defaultdict(dict)
                metrics_dict[task]['instruction_metrics'][inst_name]['num_datapoints'] = len(task_instruction_specific_dict[inst_name])
                calculate_task_metrics(task, task_instruction_specific_dict[inst_name], metrics_dict[task]['instruction_metrics'][inst_name],instruction_level=True)



    metric_output_file_name = args.outputfile.split('.')[-2] + '_metrics.json'
    with open(metric_output_file_name, 'w') as metric_output_file:
        json.dump(metrics_dict, metric_output_file, indent=4) 
        



if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)

    calculate_metrics(args)


#python -m scripts.create_data_seq2seq --outputfile scripts/seq2seqfiles/traindata_tasks1.json
