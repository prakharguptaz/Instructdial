parlai display_data -t bot_adversarial_dialogue  --bad-include-persona True --datapath ./bad
parlai display_data -t bot_adversarial_dialogue:HumanSafetyEvaluation  --bad-include-persona True --datapath ./bad
parlai display_data -t  sensitive_topics_evaluation -dt valid --datapath ./bad
