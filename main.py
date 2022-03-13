import argparse
import logging
import random
from datetime import datetime

import numpy as np
import torch


def init_logger(date):
    logging.basicConfig(
        filename='logs/{}.log'.format(date),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def print_config(config, logger):
    logger.info("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (25 - len(key)))
        logger.info("{} -->   {}".format(keystr, val))
    logger.info("**************** MODEL CONFIGURATION ****************")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(seed, args, logger):
    set_random_seed(seed)
    print_config(args, logger)

    if args['task'] == 'wsc':
        if args['expbert']:
            raise NotImplementedError
        else:
            if args["fast"] == "ours":
                from train_fast_ours import WinogradTrainer
            elif args["fast"] == "random":
                from train_fast_random import WinogradTrainer
            else:
                if args["commonsense"]:
                    from train_ok_transformer import WinogradTrainer
                else:
                    from train_transformer import WinogradTrainer
            trainer = WinogradTrainer(args, logger)
    else:
        if args['expbert']:
            from train import ExpBERTGLUETrainer
            trainer = ExpBERTGLUETrainer(args)
        else:
            if args["fast"] == "ours":
                from train_fast_ours import GLUETrainer
            elif args["fast"] == "random":
                from train_fast_random import GLUETrainer
            else:
                if args["commonsense"]:
                    from train_ok_transformer import GLUETrainer
                else:
                    from train_transformer import GLUETrainer
            trainer = GLUETrainer(args, logger)
            
    accuracy = trainer.train()
    return accuracy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', type=str, default="wscr")
    parser.add_argument('--testset', type=str, default="wsc273")
    parser.add_argument('--task', type=str, default='sst-2')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--model', type=str, default="bert-base-uncased")
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--commonsense', type=bool, default=False)
    parser.add_argument('--method', type=str, default="mc")
    parser.add_argument('--gradient_checkpoint', type=bool, default=False)
    parser.add_argument('--add_pooler_output', type=bool, default=False)
    parser.add_argument('--last_pooler_output', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--cs_batch_size', type=int, default=32)
    parser.add_argument('--avg_cs_num', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--grad_clipping', type=float, default=1.0)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=20)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--evaluate_per_step', type=int, default=500)
    parser.add_argument('--evaluate_during_training', type=bool, default=False)
    parser.add_argument('--expbert', type=bool, default=False)
    parser.add_argument('--subset', type=bool, required=False)
    parser.add_argument('--fast', type=str, default="default", required=False)
    parser.add_argument('--static', type=bool, default=False, required=False)

    args = vars(parser.parse_args())

    if args["model"] == "roberta-large":
        args["cs_batch_size"] = 32
    elif args["model"] == "bert-base-uncased":
        args["cs_batch_size"] = 128

    if args['task'] in ['mrpc', 'cola', 'rte', 'sst-2', 'sts-b', 'mnli', 'qnli', 'qqp']:
        args['trainset'] = args['task'] + '-train'
        args['testset'] = args['task'] + '-dev'
        
    return args


if __name__ == '__main__':
    date = str(datetime.now())
    logger = init_logger(date)

    args = get_args()
    main(args["seed"], args, logger)
        