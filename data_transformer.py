import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


class WinogradDataset(Dataset): 
    def __init__(self, path, tokenizer, config):
        super(WinogradDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.candidates = []
        self.sentences = []
        self.labels = []

        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for ex in data["data"]:
                self.candidates.append(ex["candidates"])
                self.sentences.append(ex["sentences"])
                self.labels.append(float(ex["label"]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "candidate": self.candidates[index],
            "sentence": self.sentences[index],
            "label": self.labels[index],
        }

    def collate_fn_mask(self, batch):
        outputs = {}
        answers = []
        sentences = []
        candidates =[]
        labels = []
        index = []
        for i, ex in enumerate(batch):
            candidates.append(ex["candidate"])
            tokens = [self.tokenizer.tokenize(candidate) for candidate in ex["candidate"]]
            index.extend(len(ex["candidate"]) * [i])
            answers.extend([ex["sentence"].replace("[mask]", candidate) for candidate in ex["candidate"]])
            sentences.extend([ex["sentence"].replace("[mask]", " ".join(len(token) * [self.tokenizer.mask_token])) for token in tokens])
            labels.append(ex["label"])
        
        assert len(index) == len(sentences) == len(answers)

        encoding = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True
        )

        answers = self.tokenizer(
            answers,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )

        assert answers["input_ids"].size() == encoding["input_ids"].size()

        answers = torch.where(encoding["input_ids"] != self.tokenizer.mask_token_id, torch.ones(encoding["input_ids"].size()).long() * -100, answers["input_ids"])
        
        outputs["answers"] = answers       # Used to calculate loss   
        outputs["labels"] = torch.LongTensor(labels)
        outputs["encoding"] = encoding
        outputs["index"] = index

        return outputs

    def collate_fn_mc(self, batch):
        outputs = {}
        # answers = []
        sentences = []
        candidates =[]
        labels = []
        index = []
        for i, ex in enumerate(batch):
            candidates.append(ex["candidate"])
            # tokens = [self.tokenizer.tokenize(candidate) for candidate in ex["candidate"]]
            index.append(len(ex["candidate"]))
            # answers.extend([ex["sentnece"].replace("[mask]", candidate) for candidate in self.candidates[id]])
            sentences.extend([ex["sentence"].replace("[mask]", candidate) for candidate in ex["candidate"]])
            labels.append(ex["label"])
        
        encoding = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        outputs["labels"] = torch.LongTensor(labels)
        outputs["encoding"] = encoding
        outputs["index"] = index

        return outputs 

class SentencePairDataset(Dataset):
    def __init__(self, path, tokenizer, config):
        super(SentencePairDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.labels = []
        self.sent1ID = []
        self.sent2ID = []
        self.sentences1 = []
        self.sentences2 = []
        self.commonsense = []
        
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            for ex in data["data"]:
                if self.config["task"] == "sts-b":
                    self.labels.append(float(ex['score']))
                else:
                    self.labels.append(int(ex['label']))
                if self.config["task"] in ["mrpc", "rte", "mnli", "sts-b"]:
                    self.sentences1.append(ex['sentence1'])
                    self.sentences2.append(ex['sentence2'])
                elif self.config["task"] == "qqp":
                    self.sentences1.append(ex['question1'])
                    self.sentences2.append(ex['question2'])
                elif self.config["task"] == "qnli":
                    self.sentences1.append(ex['question'])
                    self.sentences2.append(ex['sentence'])

    def __len__(self):
        return len(self.labels)
      
    def __getitem__(self, index):
        return {
            'sentence1': self.sentences1[index],
            'sentence2': self.sentences2[index],
            'label': self.labels[index]
        }
    
    def collate_fn(self, batch):
        labels = []
        sentences1 = []
        sentences2 = []
        for ex in batch:
            labels.append(ex["label"])
            sentences1.append(ex["sentence1"])
            sentences2.append(ex["sentence2"])

        encoding = self.tokenizer(
            sentences1, sentences2,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True
        )

        encoding["labels"] = torch.LongTensor(labels) if self.config["task"] != "sts-b" else torch.FloatTensor(labels)

        return encoding
         
class SingleSentenceDataset(Dataset):
    def __init__(self, path, tokenizer, config):
        super(SingleSentenceDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.labels = []
        self.sentences = []
        
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for ex in data["data"]:
                self.labels.append(float(ex['label']))
                self.sentences.append(ex['sentence'])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        example = {
            "label": self.labels[index],
            "sentence": self.sentences[index],
        }

        return example
    
    def collate_fn(self, batch):
        labels = []
        sentences = []
        for ex in batch:
            labels.append(ex["label"])
            sentences.append(ex["sentence"])
        
        encoding = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True
        )

        encoding["labels"] = torch.LongTensor(labels)

        return encoding
