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


def merge_list(commonsense, config):
    new_list = []
    for temp in commonsense:
        new_list.extend(temp)

    new_list = list(set(new_list))
    random.shuffle(new_list)
    new_commonsense = new_list[: config["avg_cs_num"] * len(commonsense)] + [" "]
    knowledge_mask = torch.zeros(len(commonsense), len(new_commonsense))
    for i in range(len(commonsense)):
        for j in range(len(new_commonsense)):
            if new_commonsense[j] in commonsense[i]:
                knowledge_mask[i][j] = 1
    knowledge_mask[:, -1] = 1
     
    return new_commonsense, knowledge_mask


class WinogradDataset(Dataset): 
    def __init__(self, path, tokenizer, config):
        super(WinogradDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.candidates = []
        self.sentences = []
        self.commonsense = []
        self.labels = []
        self.cs2vector = {}

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            for ex in data["data"]:
                self.candidates.append(ex["candidates"])
                self.sentences.append("<knowledge> " + ex["sentences"])
                self.labels.append(int(ex["label"]))
                self.commonsense.append(ex["commonsense"])
        
        if self.config["static"]:
            self.encode_cs()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "candidate": self.candidates[index],
            "sentence": self.sentences[index],
            "commonsense": self.commonsense[index],
            "label": self.labels[index],
        }


    def encode_cs(self):
        with torch.no_grad():
            config = AutoConfig.from_pretrained(self.config["model"])
            config.output_hidden_states = True
            cs_bert = AutoModel.from_pretrained(self.config["model"], config=config).cuda().eval()
            tok = AutoTokenizer.from_pretrained(self.config["model"])
            with tqdm(total=len(self.commonsense) + 1) as pbar:
                for commonsense in self.commonsense + [[" "]]:
                    new_cs = [cs for cs in commonsense if cs not in self.cs2vector]
                    pbar.update(1)
                    if new_cs == []:
                        continue
                    else:
                        inputs = tok(new_cs, padding="longest", max_length=24, truncation=True, return_tensors="pt")
                        for k, v in inputs.items():
                            inputs[k] = v.cuda()
                        outputs = cs_bert(**inputs).hidden_states[1:]
                        for i, cs in enumerate(new_cs):
                            self.cs2vector[cs] = [outputs[j][i, 0, :].cpu() for j in range(len(outputs))]

    def collate_fn_mask(self, batch):
        outputs = {}
        answers = []
        sentences = []
        commonsenses = []
        candidates =[]
        labels = []
        index = []

        for k, ex in enumerate(batch):
            candidates.append(ex["candidate"])
            tokens = [self.tokenizer.tokenize(candidate) for candidate in ex["candidate"]]
            index.extend(len(ex["candidate"]) * [k])
            answers.extend([ex["sentence"].replace("[mask]", candidate) for candidate in ex["candidate"]])
            sentences.extend([ex["sentence"].replace("[mask]", " ".join(len(token) * [self.tokenizer.mask_token])) for token in tokens])
            for candidate in ex["candidate"]:
                commonsenses.append(ex["commonsense"])
            labels.append(ex["label"])
        
        assert len(index) == len(sentences) == len(answers)

        encoding1 = self.tokenizer(
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

        assert answers["input_ids"].size() == encoding1["input_ids"].size()

        answers = torch.where(encoding1["input_ids"] != self.tokenizer.mask_token_id, torch.ones(encoding1["input_ids"].size()).long() * -100, answers["input_ids"])
        
        outputs["answers"] = answers       # Used to calculate loss   
        outputs["labels"] = torch.LongTensor(labels)
        outputs["encoding"] = encoding1
        outputs["index"] = index


        merge_commonsenses, knowledge_mask = merge_list(commonsenses, self.config)
        if self.config["static"]:
            knowledge = [[] for i in range(len(self.cs2vector[" "]))]
            for cs in merge_commonsenses:
                for i in range(len(knowledge)):
                    knowledge[i].append(self.cs2vector[cs][i].tolist())
            
            outputs["knowledge"] = torch.tensor(knowledge)
        else:
            times = int(len(merge_commonsenses) / self.config["cs_batch_size"]) + 1
            outputs["commonsense"] = []
            outputs["cs_sent"] = merge_commonsenses
            for i in range(times):
                if len(merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]]) != 0:
                    outputs["commonsense"].append(
                            self.tokenizer(
                                merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]],
                                add_special_tokens=True,
                                padding="longest",
                                truncation=True,
                                max_length=24,  # TODO: add to argument
                                return_tensors="pt",
                                return_attention_mask=True,
                                return_token_type_ids=True
                        )
                    )

        outputs["knowledge_mask"] = knowledge_mask

        return outputs

    def collate_fn_mc(self, batch):
        outputs = {}
        sentences = []
        commonsenses = []
        candidates =[]
        labels = []
        index = []
        index2 = []
        for k, ex in enumerate(batch):
            candidates.append(ex["candidate"])
            index.append(len(ex["candidate"]))
            index2.extend(len(ex["candidate"]) * [k])
            sentences.extend([ex["sentence"].replace("[mask]", candidate) for candidate in ex["candidate"]])
            for candidate in ex["candidate"]:
                commonsenses.append(ex["commonsense"])
            labels.append(ex["label"])

        encoding1 = self.tokenizer(
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
        outputs["encoding"] = encoding1
        outputs["index"] = index

        merge_commonsenses, knowledge_mask = merge_list(commonsenses, self.config)
        if self.config["static"]:
            knowledge = [[] for i in range(len(self.cs2vector[" "]))]
            for cs in merge_commonsenses:
                for i in range(len(knowledge)):
                    knowledge[i].append(self.cs2vector[cs][i].tolist())
            
            outputs["knowledge"] = torch.tensor(knowledge)
        else:
            times = int(len(merge_commonsenses) / self.config["cs_batch_size"]) + 1
            outputs["commonsense"] = []
            outputs["cs_sent"] = merge_commonsenses
            for i in range(times):
                if len(merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]]) != 0:
                    outputs["commonsense"].append(
                            self.tokenizer(
                                merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]],
                                add_special_tokens=True,
                                padding="longest",
                                truncation=True,
                                max_length=24,  # TODO: add to argument
                                return_tensors="pt",
                                return_attention_mask=True,
                                return_token_type_ids=True
                        )
                    )

        outputs["knowledge_mask"] = knowledge_mask
        
        return outputs      


class SentencePairDataset(Dataset):
    def __init__(self, path, tokenizer, config):
        super(SentencePairDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.labels = []
        self.sentences1 = []
        self.sentences2 = []
        self.commonsense = []
        self.cs2vector = {}
        
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            for ex in data["data"]:
                if self.config["task"] == "sts-b":
                    self.labels.append(float(ex["score"]))
                else:
                    self.labels.append(int(ex["label"]))

                if self.config["task"] in ["mrpc", "rte", "mnli", "sts-b"]:
                    self.sentences1.append("<knowledge> " + ex["sentence1"])
                    self.sentences2.append(ex["sentence2"])
                elif self.config["task"] == "qqp":
                    self.sentences1.append("<knowledge> " + ex["question1"])
                    self.sentences2.append(ex["question2"])
                elif self.config["task"] == "qnli":
                    self.sentences1.append("<knowledge> " + ex["question"])
                    self.sentences2.append(ex["sentence"])

                self.commonsense.append(ex["commonsense"])
        
        if self.config["static"]:
            self.encode_cs()

    def __len__(self):
        return len(self.labels)
      
    def __getitem__(self, index):
        return {
            "sentence1": self.sentences1[index],
            "sentence2": self.sentences2[index],
            "commonsense": self.commonsense[index],
            "label": self.labels[index]
        }
    
    def encode_cs(self):
        with torch.no_grad():
            config = AutoConfig.from_pretrained(self.config["model"])
            config.output_hidden_states = True
            cs_bert = AutoModel.from_pretrained(self.config["model"], config=config).cuda().eval()
            tok = AutoTokenizer.from_pretrained(self.config["model"])
            with tqdm(total=len(self.commonsense) + 1) as pbar:
                for commonsense in self.commonsense + [[" "]]:
                    new_cs = [cs for cs in commonsense if cs not in self.cs2vector]
                    pbar.update(1)
                    if new_cs == []:
                        continue
                    else:
                        inputs = tok(new_cs, padding="longest", max_length=24, truncation=True, return_tensors="pt")
                        for k, v in inputs.items():
                            inputs[k] = v.cuda()
                        outputs = cs_bert(**inputs).hidden_states[1:]
                        for i, cs in enumerate(new_cs):
                            self.cs2vector[cs] = [outputs[j][i, 0, :].cpu() for j in range(len(outputs))]

    def collate_fn(self, batch):
        encoding = {}
        sentence1s = []
        sentence2s = []
        commonsenses = []
        labels = []
        for ex in batch:
            sentence1s.append(ex["sentence1"])
            sentence2s.append(ex["sentence2"])
            commonsenses.append(ex["commonsense"])
            labels.append(ex["label"])
        
        encoding["encoding1"] = self.tokenizer(
            sentence1s, sentence2s,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        merge_commonsenses, knowledge_mask = merge_list(commonsenses, self.config)
        if self.config["static"]:
            knowledge = [[] for i in range(len(self.cs2vector[" "]))]
            for cs in merge_commonsenses:
                for i in range(len(knowledge)):
                    knowledge[i].append(self.cs2vector[cs][i].tolist())
            
            encoding["knowledge"] = torch.tensor(knowledge)
        else:
            times = int(len(merge_commonsenses) / self.config["cs_batch_size"]) + 1
            encoding["commonsense"] = []
            encoding["cs_sent"] = merge_commonsenses
            for i in range(times):
                if len(merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]]) != 0:
                    encoding["commonsense"].append(
                            self.tokenizer(
                                merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]],
                                add_special_tokens=True,
                                padding="longest",
                                truncation=True,
                                max_length=24,  # TODO: add to argument
                                return_tensors="pt",
                                return_attention_mask=True,
                                return_token_type_ids=True
                        )
                    )

        encoding["labels"] = torch.LongTensor(labels) if self.config["task"] != "sts-b" else torch.FloatTensor(labels)
        encoding["knowledge_mask"] = knowledge_mask
        
        return encoding
    
    def create_cs_inputs(self):
        encoding = {}
        merge_commonsenses, knowledge_mask = merge_list(self.commonsense, self.config, None, None)
        times = int(len(merge_commonsenses) / self.config["cs_batch_size"]) + 1
        encoding["commonsense"] = []
        encoding["cs_sent"] = []
        for i in range(times):
            if len(merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]]) != 0:
                encoding["commonsense"].append(
                        self.tokenizer(
                            merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]],
                            add_special_tokens=True,
                            padding="longest",
                            truncation=True,
                            max_length=24,  # TODO: add to argument
                            return_tensors="pt",
                            return_attention_mask=True,
                            return_token_type_ids=True
                    )
                )
                encoding["cs_sent"].append(merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]])
        
        cs2embedding = {}
        for i, cs in enumerate(merge_commonsenses):
            cs2embedding[cs] = i
        
        return encoding, cs2embedding

class SingleSentenceDataset(Dataset):
    def __init__(self, path, tokenizer, config):
        super(SingleSentenceDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.labels = []
        self.sentences = []
        self.commonsense = []
        self.cs2vector = {}

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            for ex in data["data"]:
                self.labels.append(int(ex["label"]))
                self.sentences.append("<knowledge> " + ex["sentence"])
                self.commonsense.append(ex["commonsense"])
        
        if self.config["static"]:
            self.encode_cs()
        
    def __len__(self):
        return len(self.labels)
      
    def __getitem__(self, index):
        return {
            "sentence": self.sentences[index],
            "commonsense": self.commonsense[index],
            "label": self.labels[index]
        }

    def encode_cs(self):
        with torch.no_grad():
            config = AutoConfig.from_pretrained(self.config["model"])
            config.output_hidden_states = True
            cs_bert = AutoModel.from_pretrained(self.config["model"], config=config).cuda().eval()
            tok = AutoTokenizer.from_pretrained(self.config["model"])
            with tqdm(total=len(self.commonsense) + 1) as pbar:
                for commonsense in self.commonsense + [[" "]]:
                    new_cs = [cs for cs in commonsense if cs not in self.cs2vector]
                    pbar.update(1)
                    if new_cs == []:
                        continue
                    else:
                        inputs = tok(new_cs, padding="longest", max_length=24, truncation=True, return_tensors="pt")
                        for k, v in inputs.items():
                            inputs[k] = v.cuda()
                        outputs = cs_bert(**inputs).hidden_states[1:]
                        for i, cs in enumerate(new_cs):
                            self.cs2vector[cs] = [outputs[j][i, 0, :].cpu() for j in range(len(outputs))]
    
    def collate_fn(self, batch):
        encoding = {}
        sentences = []
        commonsenses = []
        labels = []
        for ex in batch:
            sentences.append(ex["sentence"])
            commonsenses.append(ex["commonsense"])
            labels.append(ex["label"])
 
        encoding["encoding1"] = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        merge_commonsenses, knowledge_mask = merge_list(commonsenses, self.config)
        if self.config["static"]:
            knowledge = [[] for i in range(len(self.cs2vector[" "]))]
            for cs in merge_commonsenses:
                for i in range(len(knowledge)):
                    knowledge[i].append(self.cs2vector[cs][i].tolist())
            
            encoding["knowledge"] = torch.tensor(knowledge)
        else:
            times = int(len(merge_commonsenses) / self.config["cs_batch_size"]) + 1
            encoding["commonsense"] = []
            encoding["cs_sent"] = merge_commonsenses
            for i in range(times):
                if len(merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]]) != 0:
                    encoding["commonsense"].append(
                            self.tokenizer(
                                merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]],
                                add_special_tokens=True,
                                padding="longest",
                                truncation=True,
                                max_length=24,  # TODO: add to argument
                                return_tensors="pt",
                                return_attention_mask=True,
                                return_token_type_ids=True
                        )
                    )

        encoding["labels"] = torch.LongTensor(labels)
        encoding["knowledge_mask"] = knowledge_mask
        
        return encoding