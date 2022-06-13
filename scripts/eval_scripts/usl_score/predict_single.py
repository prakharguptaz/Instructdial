from collections import namedtuple
from datasets import VUPDataset, NUPDataset, MLMDataset
from models.VUPScorer import VUPScorer
from models.NUPScorer import NUPScorer
from models.MLMScorer import MLMScorer
import argparse
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(args, X_test, metric):
    '''
    Predict the score for each instance.
    Return an array of scores
    '''
    if metric == "VUP":
        model = VUPScorer.load_from_checkpoint(checkpoint_path=args.weight_path)

    elif metric == "NUP":
        model = NUPScorer.load_from_checkpoint(checkpoint_path=args.weight_path)

    elif metric == "MLM":
        model = MLMScorer.load_from_checkpoint(checkpoint_path=args.weight_path)

    else:
        raise Exception('Please select model from the following. VUP|NUP|MLM')

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        scores = []
        for x in X_test:
            if isinstance(x, str): # has a single string
                score = model.predict(x, normalize=args.normalize)
            else: # otherwise, a tuple of (c,r)
                score = model.predict(*x)
            scores.append(score)
        return scores


def read_dataset(path):
    '''
    - A line of sentence x -> Output a of sentence x
    - A line of 2 sentences separated by tab -> a tuple of (c,r)
    '''
    arr = []
    with open(path) as f:
        for line in f:
            sents = [ x.strip() for x in line.split('\t') ]
            if len(sents) == 1:
                arr.append(sents[0])
            else:
                arr.append(sents)
        f.close()
    return arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='USL-H inference script')
    parser.add_argument('--metric', type=str, required=True, help='Choose a metric to train. VUP|NUP|MLM')
    parser.add_argument('--weight-path', type=str, default='./checkpoints', help='Path to directory that stores the weight')
    parser.add_argument('--normalize', action='store_true', help='option for MLM whether to do normalization or not')
    parser.add_argument('--test-path', type=str, required=True, help='Path to the directory of testing set')

    args = parser.parse_args()

    test_data = read_dataset(args.test_path)
    scores = predict(args, test_data, args.metric)
    print (scores)
