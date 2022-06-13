for i in $(seq -f "%03g" 1 34)
do
    wget -P ./chitchat/test https://raw.githubusercontent.com/facebookresearch/accentor/main/v1.0/accentor-sgd/test/dialogues_${i}.json
done

for i in $(seq -f "%03g" 1 127)
do
    wget -P ./chitchat/train https://raw.githubusercontent.com/facebookresearch/accentor/main/v1.0/accentor-sgd/train/dialogues_${i}.json
done

for i in $(seq -f "%03g" 1 20)
do
    wget -P ./chitchat/dev https://raw.githubusercontent.com/facebookresearch/accentor/main/v1.0/accentor-sgd/dev/dialogues_${i}.json
done