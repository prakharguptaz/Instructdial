source /usr0/home/yitingye/miniconda3/etc/profile.d/conda.sh

cd usl_score

conda activate usl_eval
python predict.py \
    --weight-dir dailydialog/ \
    --context-file datasets/contexts.txt \
    --response-file datasets/hyps.txt \
    --output-score datasets/score.json 