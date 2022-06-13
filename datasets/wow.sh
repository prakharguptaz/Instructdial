wget -P ./ -nc http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
tar --keep-old-files  -xvzf ./wizard_of_wikipedia.tgz
python ./wizard_generator.py
printf '%s\n' "${PWD##*/}"