import argparse
import json
from scorer import Scorer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='USL-H inference script')
    parser.add_argument('--weight-dir', type=str, required=True, help='Path to directory that stores the weight')
    parser.add_argument('--context-file', type=str, required=True, help='Path to context file. Each line is a context.')
    parser.add_argument('--response-file', type=str, required=True, help='Path to response file. Each line is a response.')
    parser.add_argument('--output-score', type=str, default='output_scores.json', help='Path to the score output')

    args = parser.parse_args()
    scorer = Scorer(args)

    contexts = []
    responses = []
    with open(args.context_file) as f:
       for line in f:
           contexts.append(line)
       f.close()
    with open(args.response_file) as f:
       for line in f:
           responses.append(line)
       f.close()

    avg_score, scores = scorer.get_scores(contexts, responses, normalize=True)
    print (avg_score)

    with open(args.output_score, 'w') as f:
        for score in scores:
            json_text = json.dumps(score)
            f.write(json_text + '\n')
        f.close()
    print (f'[!] evaluation complete. output to {args.output_score}')
