# InstructDial: Improving Zero and Few-shot Generalization in Dialogue through Instruction Tuning

Code for the paper Code for the paper InstructDial: Improving Zero and Few-shot Generalization in Dialogue through Instruction Tuning - [Link](https://arxiv.org/abs/2205.12673)

## Overview
Instruction tuning is an emergent paradigm in NLP wherein natural language instructions are leveraged with language models to induce zero-shot performance on unseen tasks. Instructions have been shown to enable good performance on unseen tasks and datasets in both large and small language models. Dialogue is an especially interesting area to explore instruction tuning because dialogue systems perform multiple kinds of tasks related to language (e.g., natural language understanding and generation, domain-specific interaction), yet instruction tuning has not been systematically explored for dialogue-related tasks. We introduce InstructDial, an instruction tuning framework for dialogue, which consists of a repository of 48 diverse dialogue tasks in a unified text-to-text format created from 59 openly available dialogue datasets. Next, we explore cross-task generalization ability on models tuned on InstructDial across diverse dialogue tasks. Our analysis reveals that InstructDial enables good zero-shot performance on unseen datasets and tasks such as dialogue evaluation and intent detection, and even better performance in a few-shot setting. To ensure that models adhere to instructions, we introduce novel meta-tasks. We establish benchmark zero-shot and few-shot performance of models trained using the proposed framework on multiple dialogue tasks.

<img src="/images/intro.jpg" width="400" height="600">

### InstructDial Description
InstructDial contains a collection of dialogue datasets transformed into one or more into dialogue tasks. For every dataset, there exist a bash script in the ```datasets``` folder that downloads and extracts the dataset from open sources, along with a dataset reader script in the ```data_utils``` folder that formats the raw dataset into a format that makes it possible to plug in the dataset into a new task. Each dialogue task (such as keywood based response generation) can use one or more dialogue datasets. The config for each task is specified through a json file (example file ```configs\config_tasks1```). The config file contains the list of datasets included in the task, along with some hyperparameters. Finally, the instances from the tasks are converted into seq2seq format for tuning a language model. This procedure is shown in the figure below. We describe each step in more detail below.

Note: We are open to incorporating new datasets and tasks in this repo on request (through github issues). Otherwise, one can fork this repo and add new tasks in their private repo.

<img src="/images/instructdialoverview.jpg" width="400" height="450">



## Adding datasets

For each dataset, all download and preprocessing scripts are present in the ```datasets``` folder. Please add a new bash script to process data to add a new dataset. The ```download_datasets.sh``` runs the bash scripts for all datasets. 
Some datasets need extra steps for setup. For example, for dialoglue data, you will bneed to run the scripts described in their readme to download the dataset. 

## Dataset readers

Every dataset needs a config in a config file (such as ```config/sample_config_tasks.json``` config file) for hyperparameters, file locations, split information, etc. However, most datareaders have a default config defined in their corresponding datareader file. Here is a sample command to test a datareader for coqa dataset. The test function prints the first 5 lines of the dataset.
```
python run.py --configfile configs/sample_config_tasks.json --dataset coqa
```

## Task files

Every task needs a config in the configs folder such as (such as ```config/sample_config_tasks.json``` config file) file for dataset readers to use, instruction module to use, hyperparas, file locations, split information, etc. Here is a sample command to test a datareader for question_generation. It saves the output in a folder. Default argument for number of datapoints is set to 10.
```
python run_tasks.py --configfile configs/sample_config_tasks.json --task question_generation --tasks_output_folder tasks_files/$TASK_FOLDER/ --max_data 200
```

The config file can be used to specify the instances of which datasets should be included in this task. Separate config files should be maintained for creating data for train and test the tasks. Each config json contains a sub-config for the datasets which includes which split to use for that dataset. If the either a config line for thet task or the datasets involve for the task is not found in the specified config file, the code will throw an error.

### Creating task files for multiple tasks 
To generate task file outputs in bulk, follow this example
```
create_tasks_files.sh
```

## Creating seq2seq files from task files

The task files contain only the innput and output instances. The following file formats the instances into a prompt by concatenating instructions, post prompts etc in the input. 

This file also adds the meta tasks (instruction selection and instruction prediction) and the None-of-the-above option to the final data.

Generate data formatted for seq2seq training using the config file at configs/$SEQ2SEQ_TASK_CONFIG.json below
For generating training datasets
```
python -m scripts.create_data_text2text --outputfile scripts/text2textfiles/$OUTFILE --configfile configs/$SEQ2SEQ_TASK_CONFIG --tasksfiles_folder tasks_files/$TASK_FOLDER/  --max_task_size $NUMBER --max_data $MAX_DATA  --none_of_above_prob $PROBN --instruction_option_size $NUMBER --instruction_binary_size $NUMBER
```
For this example command you can use ```$OUTFILE=sample_seqfile.json  $SEQ2SEQ_TASK_CONFIG=sample_experiment.json $TASK_FOLDER=tasks_files/tasks_files-full-trainconfig1/ $PROBN=0.1``` For each instance, the input is saved in the prompt field and output in all_outputs field.

For generating test datasets (no meta and nota data is created)
```
python -m scripts.create_data_text2text --outputfile scripts/text2textfiles/$OUTFILE --configfile configs/$SEQ2SEQ_TASK_CONFIG --tasksfiles_folder tasks_files/$TASK_FOLDER/  --max_task_size $NUMBER --max_data $MAX_DATA --instruction_option_size -1 --instruction_binary_size -1
```

Description of keys and values in a seq2seq config file 

```js
{
  //list of tasks
  "task-files": [
    "answer_selection" 
  ],
  //list of datasets to be excluded from all tasks
  "datasets_excluded":[
    "cider"
  ],
  //at task-level set datasets to include an exclude (both optional, read below)
  "task_datasets_details":{
    "answer_selection":{
	 //	If the optional key datasets-included is set, only the datasets in this list will be used for this task. 
	 // They also necessary to be present in the task data
      "datasets-included":["coqa", "quac", "cider", "mutual", "timedial"],
	 // If the optional key datasets-excluded is set, these datasets will be excluded from only this task.
	 // A dataset set here will be excluded even if it is present in datasets-included
      "datasets-excluded":["coqa", "quac", "cider", "mutual"]
    }
  },
  "few_shot_tasks": {
	// to include few shot training examples for a task
	// set the properties (task_datasets_details) for these tasks in the fields above
	"intent_classification":{
      "k-shot": 100,
      "data-dist": "uniform"
    },
  
  }
}
```

## Training model using the seq2seq files
Following scipts need the latest version of deepspeed to run.
Set the train and validation files in the bash file below

For traning a Bart-large type model (Need machines with atleast two GPUs)
```bash
bash scripts/train-idb0.sh
```

For training a T0-3B type model (needs machines with two GPUs, both greater than 40Gb in size)
```bash
bash scripts/train-idt0.sh
```

## Generate model outputs and save to file
```bash
python run_generate.py --output_prefix PREFIX_FORFILE --input_file INPUT_FILE --model CHECKPOINT --batch_size 10
```
PREFIX_FORFILE can be set to any string or empty, INPUT_FILE should be the test file

Run this script for probability generation for ```yes``` token for the dialogue evaluation task
```bash
python run_prob_generate.py --output_prefix PREFIX_FORFILE --input_file INPUT_FILE --model CHECKPOINT --batch_size 10
```

## Running eval on model outputs and save to file (output to same location after appending _metrics to file name )
```
python run_eval.py --outputfile OUTPUT_FILE
```
Please read the README in the folder ```scripts/eval_scripts/instr_data/``` to use additional automated metrics


## Summary of steps above
* To add a dataset d, add a bash file in datasets folder that will download and extract the dataset d in the datasets folder.
* Add a dataset reader script in the data_util folder that converts the raw dataset to a format that can be used by the instruction creation scripts. For example, for a knowledge grounded generation task, the datareader script should expose a field for each instance that contains the knowledge text. Run ```run_tasks.py``` to check the if the data is created correctly.
* Add the datareader to the datareaders.py file.
* To add a new task, create a new file in the intructions folder. You can start from a copy of any existing task in that folder.
* Add an entry for the task to the task config file. Add all relevant datasets in the datasets list of that entry. Add the name of the instruction file you created in the instruction_files. 
* If the datasets used for that task is not present in the ```dataset_configs``` key of the config, add a new entry for the dataset.
* Run run_tasks.py everytime you change the config for a task or add new datasets for the task.
* Run create_data_text2text to create the final dataset conatining tasks specified inthe experiment config, or everytime you change anyfile or task created in the steps above.

**Note** that you can change the formatting used for seq2seq data preparation by changing variables in the constants.py and utils folder.


## To-do
Will soon release a model trained on all tasks and some prepared training data