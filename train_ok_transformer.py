import argparse
import json
import logging
import os
import random
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
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from data_ok_transformer import (
    WinogradDataset,
    SingleSentenceDataset,
    SentencePairDataset,
)
from model_ok_transformer import GLUEModel, WinogradModel

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


def acc_and_f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
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
    }


TASK2PATH = {
    "wsc273": "data/wsc273/wsc273-atomic2020-5.json",
    "wscr": "data/wscr/wscr-atomic2020-5.json",
    "winogrande": "data/winogrande/winogrande-atomic2020-5.json",
    "winogrande-train": "data/winogrande/winogrande-train-atomic2020-5.json",
    "winogrande-dev": "data/winogrande/winogrande-dev-atomic2020-5.json",
    "winogender": "data/winogender/winogender-atomic2020-5.json",
    "pdp": "data/pdp/pdp-atomic2020-5.json",
    "mrpc-train": 'data/glue/MRPC/train-atomic2020-5.json',
    "mrpc-dev": "data/glue/MRPC/dev-atomic2020-5.json",
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


class WinogradTrainer:
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config

        print("Load Model Begin.")        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        self.model = WinogradModel(self.config).cuda()
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<knowledge>"]})
        self.model.resize_token_embeddings(len(self.tokenizer))
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
        print("Prepare Inputs")

        t_total = int(len(self.train_loader) * self.config["epochs"])

        self.init_optimizer()
        self.init_scheduler(t_total)
        self.record_accuracy = defaultdict(dict)

    def init_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config["learning_rate"])

    def init_scheduler(self, t_total):
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.config['warmup_proportion'] * t_total),
            num_training_steps=t_total)

    def train(self):
        self.test(-1)
        for e in range(self.config["epochs"]):
            self.model.train()
            self.model.zero_grad()
            with tqdm(total=len(self.train_loader), ncols=140) as pbar:
                labels = []
                predictions = []
                loss = []
                for step, examples in enumerate(self.train_loader):
                    examples = examples.copy()
                    labels.extend(examples["labels"].tolist())
                    outputs = self.model(examples)
                    predictions.extend(outputs["prediction"].tolist())
                    loss.append(outputs["loss"].item())
                    outputs["loss"].backward()
                    if not self.config["static"]:
                        knowledge_grad = [temp.grad.detach() for temp in outputs["knowledge"]]
                        self.model.update_cs(examples, knowledge_grad)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clipping'])
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()

                    if step != 0 and step % self.config['evaluate_per_step'] == 0 and self.config['evaluate_during_training']:
                        self.logger.info("=================== EPOCH {} ===================".format(e))
                        self.logger.info("Training Loss: {}".format(np.mean(loss)))
                        self.test(e)
                        self.model.train()
                    pbar.update(1)

            evaluations = self.compute(predictions, labels)
            self.logger.info("==================================== EPOCH {} ====================================".format(e))
            self.logger.info("Training Accuracy: {}.".format(evaluations))
            self.logger.info("Training Loss: {}".format(np.mean(loss)))
            self.logger.info("Number of Training data: {}".format(len(labels)))
            
            self.test(e)
    
        return self.record_accuracy

    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for i, test_loader in enumerate(self.test_loaders):
                with tqdm(total=len(test_loader), ncols=140) as pbar:
                    loss = []
                    labels = []
                    predictions = []
                    for step, examples in enumerate(test_loader):
                        examples = examples.copy()
                        labels.extend(examples["labels"].tolist())
                        outputs = self.model(examples)
                        predictions.extend(outputs["prediction"].tolist())

                        pbar.update(1)

                    evaluations = self.compute(predictions, labels)
                    for k in evaluations.keys():
                        if k not in self.record_accuracy[self.testsets[i]]:
                            self.record_accuracy[self.testsets[i]][k] = 0.

                        if evaluations[k] > self.record_accuracy[self.testsets[i]][k]:
                            self.record_accuracy[self.testsets[i]][k] = evaluations[k]
                            
                self.logger.info("[EPOCH {}] Accuracy: {}. (Number of Data {})".format(epoch, evaluations, len(predictions)))

    def compute(self, predictions, references):
        return {"accuracy": simple_accuracy(predictions, references)}


class GLUETrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        print("Load Model Begin.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        self.model = GLUEModel(self.config).cuda()
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<knowledge>"]})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.original = AutoModel.from_pretrained(self.config["model"]).eval().cuda()
        print("Load Model Completed.")


        print("Load Dataset Begin.")
        self.train_dataset = TASK2DATASET[self.config['task']](  #TODO: 
            TASK2PATH[self.config['trainset']],
            self.tokenizer, self.config
            )

        logger.info("Train Dataset Path {}.".format(TASK2PATH[self.config['trainset']]))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            collate_fn=self.train_dataset.collate_fn,
            shuffle=self.config["shuffle"],
        )

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

        print("Prepare Inputs")

        # self.lengths = []
        self.train_iterator = []
        self.test_iterator = []
        with tqdm(total=len(self.train_loader)) as pbar:
            for examples in self.train_loader:
                # self.lengths.append(len(examples["cs_sent"]))
                self.train_iterator.append(examples)
                pbar.update(1)

        with tqdm(total=len(self.test_loader)) as pbar:
            for examples in self.test_loader:
                # self.lengths.append(len(examples["cs_sent"]))
                self.test_iterator.append(examples) 
                pbar.update(1)
        # print("avg cs num per batch: {}".format(np.mean(self.lengths)))
        # raise ValueError
        t_total = len(self.train_loader) * self.config["epochs"]

        self.init_optimizer()
        self.init_scheduler(t_total)
        self.record_accuracy = {}

    def init_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config["learning_rate"])

    def init_scheduler(self, t_total):
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.config['warmup_proportion'] * t_total),
            num_training_steps=t_total)

    def train(self):
        self.test(-1)
        for e in range(self.config["epochs"]):
            self.model.train()
            self.model.zero_grad()
            with tqdm(total=len(self.train_loader), ncols=140) as pbar:
                labels = []
                predictions = []
                loss = []
                for step, examples in enumerate(self.train_loader):
                    examples = examples.copy()
                    labels.extend(examples["labels"].tolist())
                    outputs = self.model(examples)
                    predictions.extend(outputs["prediction"].tolist() if not isinstance(outputs["prediction"], list) else outputs["prediction"])
                    loss.append(outputs["loss"].item())
                    outputs["loss"].backward()
                    if not self.config["static"]:
                        knowledge_grad = [temp.grad.detach() for temp in outputs["knowledge"]]
                        self.model.update_cs(examples, knowledge_grad)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clipping'])
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
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
                    examples = examples.copy()
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

            assert len(labels) == len(predictions)
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
