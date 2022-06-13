#!/bin/bash

set -e
OUTPUT_FOLDER=tasks_files-full-train
TaskArray=('act_classification' 'document_grounded_generation' 'intent_present' 'relation_classification' 'act_generation' 'emotion_generation' 'keyword_controlled_generation' 'relation_present' 'advice_generation' 'emotion_tagging' 'knowledge_grounded_generation' 'response_generation' 'advice_present' 'endswith_controlled_generation' 'nli_classification' 'schema_based_generation' 'answer_generation' 'eval_binary' 'persona_grounded_generation' 'slot_present' 'answer_selection' 'eval_ranking' 'persuasion_generation' 'slot_tagging' 'beginswith_controlled_generation' 'eval_rating' 'persuasion_present' 'summarization' 'belief_state_generation' 'persuasion_strategy' 'target_controlled_generation' 'db_based_generation' 'graph_based_generation' 'deal_present' 'intent_classification' 'question_generation')
for val1 in ${TaskArray[*]}; do
     echo $val1
	 python run_tasks.py --configfile configs/config_tasks1.json --task $val1 --max_data 100000000 --tasks_output_folder tasks_files/$OUTPUT_FOLDER
#	 python run_tasks.py --configfile configs/config_tasks1.json --task $val1 --data_sample_type max --tasks_output_folder tasks_files/$OUTPUT_FOLDER
	 echo $'\n'
done
