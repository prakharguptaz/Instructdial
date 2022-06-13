from pathlib import Path
import json
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

def load_instr_data(base_dir, input_file):
    if input_file is None or input_file=='':
        print('SPECIFY INPUT FILEPATH --input_file, EXITING')
        exit(0)
    # file = Path(f'{base_dir}/{input_file}')
    file = Path(input_file)
    
    contexts, responses, references, scores = [], [], [], []
    response_per_sample = []

    with file.open() as f:
        idx = 0
        for line in f.readlines():
            dp = json.loads(line)
            get_responsesfromdict(dp)
            
            for response in dp['response_list']:
                contexts.append(dp['dialogue_context'])
                responses.append(response)
                references.append('NO REF')
                scores.append(0)
                
                idx += 1
            response_per_sample.append(idx)

    return {
        'contexts': contexts,
        'responses': responses,
        'references': references,
        'scores': scores,
        'response_per_sample': response_per_sample
    }

if __name__ == '__main__':
    data = load_instr_data('.')
    with open('instr_data.json', 'w') as f:
        json.dump(data, f)

