import argparse
import os
import json

from data.grade_data.data_loader import load_grade_data
from data.usr_data.data_loader import load_usr_data
from data.dstc6_data.data_loader import load_dstc6_data
from data.fed_data.data_loader import load_fed_data, load_fed_dialog_data
from data.dstc9_data.data_loader import load_dstc9_data
from data.holistic_data.data_loader import load_holistic_data
from data.engage_data.data_loader import load_engage_data
# from data.dstc10_data.data_loader import load_dstc10_data
# from data.dstc10_eval_data.data_loader import load_dstc10_eval_data
from data.instr_data.data_loader import load_instr_data

'''
Adding new data:
from data.data.data_laoder import load_new_data # the customized data loader function
'''

from maude.data_parser import gen_maude_data
from grade.data_parser import gen_grade_data
from ruber.data_parser import gen_ruber_data
from holistic_eval.data_parser import gen_hostilic_data
from predictive_engagement.data_parser import gen_engagement_data
from am_fm.data_parser import gen_amfm_data
from usl_dialogue_metric.data_parser import gen_usl_data
from deb.data_parser import gen_deb_data
from dynaeval.data_parser import gen_dynaeval_data
from fbd.data_parser import gen_fbd_data
from dialogrpt.data_parser import gen_dialogrpt_data

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_data', type=str, default=None)
    parser.add_argument('--target_format', type=str, default=None)
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()
    return args

def gen_baseline_data(data, data_path):
    with open(data_path, 'w') as fout:
        json.dump(data, fout)

def main(source_data, target_format, input_file=''):
    format_type = 0 # two types of generating data
    max_words = None # for models using pretrained language models with input length constraint such as BERT
    if target_format == 'maude':
        metric = 'maude'
        output_dir = f'{os.getcwd()}/maude/eval_data'
        gen_data = gen_maude_data
        suffix = '.csv'
        max_words = 500
    elif target_format == 'hostilic':
        metric = 'hostilic'
        output_dir = f'{os.getcwd()}/holistic_eval/eval_data'
        gen_data = gen_hostilic_data 
        suffix = '.csv'
        max_words = 500
    elif target_format == 'baseline':
        metric = 'baseline'
        output_dir = f'{os.getcwd()}/baseline_data'
        gen_data = gen_baseline_data
        suffix = '.json'
    elif target_format == 'usr_fed':
        metric = 'usr_fed'
        output_dir = f'{os.getcwd()}/usr_fed_data'
        gen_data = gen_baseline_data
        suffix = '.json'
        max_words = 500
    elif target_format == 'ruber':
        metric = 'ruber'
        output_dir = f'{os.getcwd()}/ruber_and_bert_ruber/RUBER/data'
        gen_data = gen_ruber_data
        suffix = ''
    elif target_format == 'bert_ruber':
        metric = 'bert_ruber'
        output_dir = f'{os.getcwd()}/PONE/PONE/data'
        gen_data = gen_ruber_data
        suffix = ''
    elif target_format == 'grade':
        metric = 'grade'
        output_dir = f'{os.getcwd()}/grade'
        gen_data = gen_grade_data
        suffix = '.json'
        format_type = 1
    elif target_format == 'predictive_engagement':
        metric = 'predictive_engagement'
        output_dir = f'{os.getcwd()}/predictive_engagement/data'
        gen_data = gen_engagement_data
        suffix = '.csv'
    elif target_format =='amfm':
        metric = 'amfm'
        output_dir = f'{os.getcwd()}/am_fm/examples/dstc6/test_data'
        gen_data = gen_amfm_data
        suffix = ''
    elif target_format == 'flowscore':
        metric = 'flowscore'
        output_dir = f'{os.getcwd()}/FlowScore/eval_data'
        gen_data = gen_baseline_data
        suffix = '.json'
    elif target_format == 'usl':
        metric = 'usl'
        output_dir = f'{os.getcwd()}/usl_dialogue_metric/usl_score/datasets'
        gen_data = gen_usl_data
        suffix = ''
    elif target_format == 'questeval':
        metric = 'questeval'
        output_dir = f'{os.getcwd()}/questeval/test_data'
        gen_data = gen_baseline_data
        suffix = '.json'
    elif target_format == 'deb':
        metric = 'deb'
        output_dir = f'{os.getcwd()}/deb/dataset'
        gen_data = gen_deb_data
        suffix = ''
    elif target_format == 'dynaeval':
        metric = 'dynaeval'
        output_dir = f'{os.getcwd()}/dynaeval/data'
        gen_data = gen_dynaeval_data
        suffix = ''
    elif target_format == 'fbd':
        metric = 'fbd'
        output_dir = f'{os.getcwd()}/fbd/datasets'
        gen_data = gen_fbd_data
        suffix = ''
    elif target_format == 'dialogrpt':
        metric = 'dialogrpt'
        output_dir = f'{os.getcwd()}/dialogrpt/test_data'
        gen_data = gen_dialogrpt_data
        suffix = '' 
    else:
        raise Exception
    '''
    Adding a new metric
    elif target_format == 'metric':
        metirc = 'METRIC_NAME'
        output_dir = 'PATH/TO/OUTPUT/DIR'
        gen_data = gen_metric_data # the customized function
        suffix = '' # the suffix of generated data
    '''
    isExist = os.path.exists(output_dir)
    if not isExist:os.makedirs(output_dir)
        

    if source_data == 'convai2_grade':
        model_names = ['bert_ranker', 'dialogGPT', 'transformer_generator', 'transformer_ranker']
        for model in model_names:
            data_path = f'{os.getcwd()}/data/grade_data'
            data = load_grade_data(data_path, 'convai2', model)
            if format_type == 0:
                output_path = f'{output_dir}/convai2_grade_{model}{suffix}'
                gen_data(data, output_path)
            elif format_type == 1:            
                gen_data(data, output_dir, f'{source_data}_{model}')

    elif source_data == 'dailydialog_grade':
        model_names = ['transformer_generator', 'transformer_ranker']
        for model in model_names:
            data_path = f'{os.getcwd()}/data/grade_data'
            data = load_grade_data(data_path, 'dailydialog', model)
            if format_type == 0:
                output_path = f'{output_dir}/dailydialog_grade_{model}{suffix}'
                gen_data(data, output_path)
            elif format_type == 1:
                gen_data(data, output_dir, f'{source_data}_{model}')

    elif source_data == 'empatheticdialogues_grade':
        model_names = ['transformer_generator', 'transformer_ranker']
        for model in model_names:
            data_path = f'{os.getcwd()}/data/grade_data'
            data = load_grade_data(data_path, 'empatheticdialogues', model)
            if format_type == 0:
                output_path = f'{output_dir}/empatheticdialogues_grade_{model}{suffix}'
                gen_data(data, output_path)
            elif format_type == 1:
                gen_data(data, output_dir, f'{source_data}_{model}')
    
    elif source_data == 'personachat_usr':
        data_path = f'{os.getcwd()}/data/usr_data'
        data = load_usr_data(data_path, 'personachat')
        
        if format_type == 0:
            output_path = f'{output_dir}/personachat_usr{suffix}'
            gen_data(data, output_path)
        elif format_type == 1:
            gen_data(data, output_dir, 'personachat_usr')
    
    elif source_data == 'topicalchat_usr':
        data_path = f'{os.getcwd()}/data/usr_data'
        data = load_usr_data(data_path, 'topicalchat')
        if format_type == 0:
            output_path = f'{output_dir}/topicalchat_usr{suffix}'
            gen_data(data, output_path)
        elif format_type == 1:
            gen_data(data, output_dir, 'topicalchat_usr')
    
    elif source_data == 'fed_dialog':
        data_path = f'{os.getcwd()}/data/fed_data'
        data = load_fed_dialog_data(data_path)
        if format_type == 0:
            output_path = f'{output_dir}/fed_dialog{suffix}'
            gen_data(data, output_path)
        elif format_type == 1:
            gen_data(data, output_dir, 'fed_dialog')
    elif source_data == 'instr':
        data_path = f'{os.getcwd()}/data/instr_data'
        data = load_instr_data(data_path, input_file=input_file)
        if format_type == 0:
            output_path = f'{output_dir}/{source_data}{suffix}'
            gen_data(data, output_path)
        elif format_type == 1:
            gen_data(data, output_dir, source_data)
    else:
        data_path = f'{os.getcwd()}/data/{source_data}_data'

        data = eval(f'load_{source_data}_data')(data_path)
        if format_type == 0:
            output_path = f'{output_dir}/{source_data}{suffix}'
            gen_data(data, output_path)
        elif format_type == 1:
            gen_data(data, output_dir, source_data)
    
    
if __name__ == '__main__':
    args = parse_args()
    all_data = ['convai2_grade', 'dailydialog_grade', 'empatheticdialogues_grade', 
                'personachat_usr', 'topicalchat_usr', 'dstc6', 'fed', 'fed_dialog', 'holistic', 'dstc9', 'engage',
                'dstc10','dstc10_eval', 'instr']

    if args.source_data is not None:
        assert args.source_data in all_data
        all_data = [args.source_data]

    if args.target_format is not None:
        metrics = [args.target_format]
    else:
        metrics = ['maude', 'hostilic', 'baseline', 'usr_fed', 'ruber', 'bert_ruber', 'grade', 'predictive_engagement', 'amfm', 'flowscore',
                  'usl', 'questeval', 'deb', 
                  'dynaeval', 
                  'dialogrpt']

    for data in all_data:
        for target in metrics:
            print(f'Generating {data} to {target}')
            main(data, target, input_file=args.input_file)
