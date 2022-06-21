wget -P ./ -nc http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
tar  -xvzf ./wizard_of_wikipedia.tgz -C WoW 

cd WoW
python ./wizard_generator.py
printf '%s\n' "${PWD##*/}"

cd ../