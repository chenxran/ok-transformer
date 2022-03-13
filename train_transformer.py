
import os

from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_transformer import (
    SingleSentenceDataset,
    SentencePairDataset,
    WinogradDataset,
)
from model_transformer import GLUEModel, WinogradModel

TASK2DATASET = {
    'mrpc': SentencePairDataset,
    'cola': SingleSentenceDataset,
    'rte': SentencePairDataset,
    'sst-2': SingleSentenceDataset,
    'qnli': SentencePairDataset,
    'mnli': SentencePairDataset,
    'sts-b': SentencePairDataset,
    'qqp':  SentencePairDataset,
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='binary'):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "accuracy": acc,
        "f1": f1,
    }


def pearson_and_spearman(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "accuracy": spearman_corr,
    }


TASK2PATH = {
    "wsc273": "data/wsc273/wsc273-atomic2020-5.json",
    "wscr": "data/wscr/wscr-atomic2020-5.json",
    "winogrande": "data/winogrande/winogrande-atomic2020-5.json",
    "winogrande-train": "data/winogrande/winogrande-train-atomic2020-5.json",
    "winogrande-dev": "data/winogrande/winogrande-dev-atomic2020-5.json",
    "winogender": "data/winogender/winogender-atomic2020-5.json",
    "pdp": "data/pdp/pdp-atomic2020-5.json",
    'mrpc-train': 'data/glue/MRPC/train-atomic2020-5.json',
    'mrpc-dev': 'data/glue/MRPC/dev-atomic2020-5.json',
    'cola-train': 'data/glue/CoLA/train-atomic2020-5.json',
    'cola-dev': 'data/glue/CoLA/dev-atomic2020-5.json',
    'rte-train': 'data/glue/RTE/train-atomic2020-5.json',
    'rte-dev': 'data/glue/RTE/dev-atomic2020-5.json',
    'sst-2-train': 'data/glue/SST-2/train-atomic2020-5.json',
    'sst-2-dev': 'data/glue/SST-2/dev-atomic2020-5.json',
    'qnli-train': 'data/glue/QNLI/train-atomic2020-5.json',
    'qnli-dev': 'data/glue/QNLI/dev-atomic2020-5.json',
    'mnli-train': 'data/glue/MNLI/train-atomic2020-5.json',
    'mnli-dev-mismatched': 'data/glue/MNLI/dev-mismatched-atomic2020-5.json',
    'mnli-dev-matched': 'data/glue/MNLI/dev-matched-atomic2020-5.json',
    'qqp-train': 'data/glue/QQP/train-atomic2020-5.json',
    'qqp-dev': 'data/glue/QQP/dev-atomic2020-5.json',
    'sts-b-train': 'data/glue/STS-B/train-atomic2020-5.json',
    'sts-b-dev': 'data/glue/STS-B/dev-atomic2020-5.json'
}


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class WinogradTrainer:
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        
        print("Load Model Begin.")        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        self.model = WinogradModel(self.config).cuda()
        print("Load Model Completed.")

        print("Load Dataset Begin.")
        logger.info("Train Dataset Path {}.".format(TASK2PATH[self.config['trainset']]))
        self.train_dataset = WinogradDataset(
            TASK2PATH[self.config["trainset"]],
            self.tokenizer, self.config
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            collate_fn=self.train_dataset.collate_fn_mask if self.config["method"] == "mask" else self.train_dataset.collate_fn_mc,
            shuffle=self.config["shuffle"],
        )

        self.testsets = self.config["testset"].split("|")
        for testset in self.testsets:
            logger.info("Test Dataset Path {}.".format(TASK2PATH[testset])) 

        self.test_datasets = [
            WinogradDataset(
                TASK2PATH[testset], 
                self.tokenizer, 
                self.config)
            for testset in self.testsets
        ]
        self.test_loaders = [
            DataLoader(
                test_dataset,
                batch_size=self.config["batch_size"],
                collate_fn=test_dataset.collate_fn_mask if self.config["method"] == "mask" else test_dataset.collate_fn_mc,
                shuffle=self.config["shuffle"],
            ) for test_dataset in self.test_datasets
        ]

        print("Load Dataset Completed.")

        t_total = len(self.train_loader) * self.config["epochs"]
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler(t_total)
        self.record_accuracy = {}

        for testset in self.testsets:
            self.record_accuracy[testset] = {}

    def init_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config["learning_rate"])

    def init_scheduler(self, t_total):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.config['warmup_proportion'] * t_total),
            num_training_steps=t_total)

    def train(self):
        self.test(-1)
        for e in range(self.config["epochs"]):
            self.model.train()
            with tqdm(total=len(self.train_loader), ncols=140) as pbar:
                labels = []
                for step, examples in enumerate(self.train_loader):
                    labels.extend(examples["labels"].tolist())
                    outputs = self.model(examples)
                    outputs["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clipping'])
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if step != 0:
                        if step % self.config['evaluate_per_step']  == 0 and self.config['evaluate_during_training']:
                            self.test(e)
                            self.model.train()
                    pbar.update(1)
    
            self.test(e)
            self.logger.info("Number of Data in Train set {}".format(len(labels)))
        
        return self.record_accuracy

    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for i, test_loader in enumerate(self.test_loaders):
                with tqdm(total=len(test_loader), ncols=140) as pbar:
                    labels = []
                    predictions = []
                    for step, examples in enumerate(test_loader):
                        outputs = self.model(examples)
                        # print(examples["labels"].tolist())
                        labels.extend(examples["labels"].tolist())
                        predictions.extend(outputs['prediction'].tolist())
                        pbar.update(1)
                
                evaluations = self.compute(predictions, labels)
                testset = self.testsets[i]
                for k in evaluations.keys():
                    if k not in self.record_accuracy[testset]:
                        self.record_accuracy[testset][k] = 0.
                    
                    if evaluations[k] > self.record_accuracy[testset][k]:
                        self.record_accuracy[testset][k] = evaluations[k]

                self.logger.info("Number of Data: {}".format(len(predictions)))

        self.logger.info("[EPOCH {}] Accuracy: {}".format(epoch, self.record_accuracy))

    def compute(self, predictions, references):
        return acc_and_f1(predictions, references, average='macro')


class GLUETrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        print("Load Model Begin.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        self.model = GLUEModel(self.config).cuda()
        print("Load Model Completed.")

        print("Load Dataset Begin.")
        logger.info("Train Dataset Path {}.".format(TASK2PATH[self.config['trainset']]))
        self.train_dataset = TASK2DATASET[self.config['task']](  #TODO: 
            TASK2PATH[self.config['trainset']],
            self.tokenizer, self.config
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            collate_fn=self.train_dataset.collate_fn,
            shuffle=self.config["shuffle"],
        )

        logger.info("Test Dataset Path {}.".format(TASK2PATH[self.config["testset"]])) 
        self.test_dataset = TASK2DATASET[self.config['task']](
            TASK2PATH[self.config["testset"]], 
            self.tokenizer, 
            self.config
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            collate_fn=self.test_dataset.collate_fn,
            shuffle=self.config["shuffle"],
        )

        print("Load Dataset Completed.")

        t_total = len(self.train_loader) * self.config["epochs"]
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler(t_total)
        self.record_accuracy = {}

    def init_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.config['weight_decay']},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters,
                                 lr=self.config["learning_rate"])

    def init_scheduler(self, t_total):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.config['warmup_proportion'] * t_total),
            num_training_steps=t_total)

    def train(self):
        self.test(-1)
        for e in range(self.config["epochs"]):
            self.model.train()
            loss = []
            labels = []
            predictions = []
            with tqdm(total=len(self.train_loader), ncols=140) as pbar:
                for step, examples in enumerate(self.train_loader):
                    labels.extend(examples["labels"].tolist())
                    outputs = self.model(examples)
                    predictions.extend(outputs["prediction"].tolist() if not isinstance(outputs["prediction"], list) else outputs["prediction"])
                    loss.append(outputs["loss"].item())
                    outputs["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clipping'])
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    pbar.update(1)

            # evaluations = self.compute(predictions, labels)
            self.logger.info("==================================== EPOCH {} ====================================".format(e))
            # self.logger.info("Training Accuracy: {}.".format(evaluations))
            self.logger.info("Training Loss: {}".format(np.mean(loss)))
            self.logger.info("Number of Training data: {}".format(len(labels)))
            self.test(e)

        return self.record_accuracy
        
    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), ncols=140) as pbar:
                labels = []
                predictions = []
                for step, examples in enumerate(self.test_loader):
                    labels.extend(examples['labels'].tolist())
                    outputs = self.model(examples)
                    predictions.extend(outputs['prediction'].tolist() if not isinstance(outputs["prediction"], list) else outputs["prediction"])

                    pbar.update(1)

            evaluations = self.compute(predictions, labels)

            for k in evaluations.keys():
                if k not in self.record_accuracy:
                    self.record_accuracy[k] = 0.

                if evaluations[k] > self.record_accuracy[k]:
                    self.record_accuracy[k] = evaluations[k]

            self.logger.info("[EPOCH {}] Dataset {} Accuracy: {}. (Number of Data {})".format(epoch, self.config["testset"], evaluations, len(predictions)))

    def compute(self, predictions, references):
        if self.config['task'] == "cola":
            return {"matthews_correlation": matthews_corrcoef(references, predictions), "accuracy": matthews_corrcoef(references, predictions)}
        elif self.config['task'] == "sts-b":
            return pearson_and_spearman(predictions, references)
        elif self.config['task'] in ["mrpc", "qqp"]:
            return acc_and_f1(predictions, references)
        elif self.config['task'] in ["sst-2", "mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli", "hans"]:
            return {"accuracy": simple_accuracy(predictions, references)}
        else:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["sst2", "mnli", "mnli_mismatched", "mnli_matched", '
                '"cola", "stsb", "mrpc", "qqp", "qnli", "rte", "wnli", "hans"]'
            )
