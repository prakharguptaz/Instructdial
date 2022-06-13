## Steps for running metrics on model outputs

* set up the environment and download the pretrained model for those metrics you want to test. For pretrained models follow instructions from https://github.com/exe1023/DialEvalMetrics
* move scripts/eval_scripts/instr_data folder to the data/ directory in the DialEvalMetrics repo
* replace/copy the files gen_data.py, eval_metric.sh, read_result.py, data_loader.py in the DialEvalMetrics repo (main folder) with those files from scripts/eval_scripts/instr_data/. Set your coda environment path in the eval_metric.sh file.
* cd to the DialEvalMetrics repo, run ```python gen_data.py --source_data instr --input_file $INPUT_FILE```
* modify DialEvalMetrics/blob/main/eval_metrics.sh and run with data instr. Set the metrics you need.
* run ```python read_result.py --metric_name $METRIC --input_file $INPUT_FILE```
* run ``python test_other_metrics.py --input_file $INPUT_FILE_WITH_SCORES```