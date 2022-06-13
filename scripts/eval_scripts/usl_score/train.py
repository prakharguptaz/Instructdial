from collections import namedtuple
from datasets import VUPDataset, NUPDataset, MLMDataset
from data_utils import read_dataset
from normalize import calc_minmax
from models.VUPScorer import VUPScorer
from models.NUPScorer import NUPScorer
from models.MLMScorer import MLMScorer

import json
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args, train_ctx, train_res, valid_ctx, valid_res):

    if args.metric == "VUP":
        train_dataset = VUPDataset(train_res, maxlen=args.res_token_len)
        valid_dataset = VUPDataset(valid_res, maxlen=args.res_token_len)
        model = VUPScorer(args).to(device)

    elif args.metric == "NUP":
        train_dataset = NUPDataset(train_ctx, train_res, ctx_token_len=args.ctx_token_len, res_token_len=args.res_token_len)
        valid_dataset = NUPDataset(valid_ctx, valid_res, ctx_token_len=args.ctx_token_len, res_token_len=args.res_token_len)
        model = NUPScorer(args).to(device)

    elif args.metric == "MLM":
       train_dataset = MLMDataset(train_res, maxlen=args.res_token_len)
       valid_dataset = MLMDataset(valid_res, maxlen=args.res_token_len)
       model = MLMScorer(args).to(device)

    else:
        raise Exception('Please select model from the following. VUP|NUP|MLM')

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    trainer = pl.Trainer(max_epochs=args.max_epochs, weights_save_path=args.weight_path, nb_sanity_val_steps=5)
    trainer.fit(model, train_dataloader, valid_dataloader)
    print ('[!] training complete')

    # Run normalization after training MLM
    if args.metric == "MLM":
        scores = calc_minmax(model, valid_res)
        with open('minmax_score.json', 'w') as f:
            f.write(json.dumps(scores, indent=4))
            f.close()
        print ('[!] mlm_minmax normalizing complete')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='USL-H training script')
    parser.add_argument('--metric', type=str, required=True, help='Choose a metric to train. VUP|NUP|MLM')
    parser.add_argument('--weight-path', type=str, default='', help='Path to directory that stores the weight')

    # Dataset
    parser.add_argument('--train-ctx-path', type=str, help='Path to context training set')
    parser.add_argument('--train-res-path', type=str, required=True, help='Path to response training set')
    parser.add_argument('--valid-ctx-path', type=str, help='Path to context validation set')
    parser.add_argument('--valid-res-path', type=str, required=True, help='Path to response validation set')
    parser.add_argument('--batch-size', type=int, default=16, help='samples per batches')
    parser.add_argument('--max-epochs', type=int, default=1, help='number of epoches to train')
    parser.add_argument('--num-workers', type=int, default=1, help='number of worker for dataset')
    parser.add_argument('--ctx-token-len', type=int, default=25, help='number of tokens for context')
    parser.add_argument('--res-token-len', type=int, default=25, help='number of tokens for response')

    # Modeling
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 regularization')

    args = parser.parse_args()

    train_ctx = read_dataset(args.train_ctx_path) if args.train_ctx_path else None
    train_res = read_dataset(args.train_res_path)
    valid_ctx = read_dataset(args.valid_ctx_path) if args.valid_ctx_path else None
    valid_res = read_dataset(args.valid_res_path)

    train(args, train_ctx, train_res, valid_ctx, valid_res)
    print ("[!] done")
