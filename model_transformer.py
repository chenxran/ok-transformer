import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig, 
    AutoModel, 
    BertModel,
    BertForMaskedLM, 
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    RobertaForMaskedLM,
)

class WinogradModel(nn.Module):
    def __init__(self, config):
        super(WinogradModel, self).__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config["model"])
        self.model_config.output_hidden_states = True
        if self.config["method"] == "mask":
            self.sent_model = AutoModelForMaskedLM.from_pretrained(config["model"], config=self.model_config)
        else:
            self.sent_model = AutoModel.from_pretrained(config["model"], config=self.model_config)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.model_config.hidden_size, 1),
            )

        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, ex):
        if self.config["method"] == 'mask':
            return self.mask(ex)
        elif self.config["method"] == 'mc':
            return self.mc(ex)
    
    def mask(self, ex):
        for k, v in ex["encoding"].items():
            ex["encoding"][k] = v.cuda()

        sent_outputs = self.sent_model(
            **ex["encoding"],
            return_dict=True
        )

        logits = sent_outputs.logits
        losses = torch.mean(self.criterion(logits.view(-1, self.model_config.vocab_size), ex["answers"].view(-1).cuda()).view(logits.size()[:2]), dim=1).squeeze()
        batch_size = len(ex["labels"].view(-1))
        temp = torch.zeros(batch_size, 5).cuda()
        # print(temp.size(), losses.size())
        old_index = -1
        j = 0
        for i, index in enumerate(ex["index"]):
            if index == old_index:
                j += 1
            else:
                j = 0
            temp[index][j] = losses[i]
            old_index = index

        loss = torch.mean(temp[torch.arange(batch_size), ex['labels']].squeeze()) + self.config['alpha'] * torch.sum(nn.functional.relu(temp[torch.arange(batch_size), ex['labels']] - temp.t() + self.config['beta']).t()) / batch_size
        prediction = torch.argmin(torch.where(temp == 0, 10000. * torch.ones(temp.size()).long().cuda(), temp), dim=1).view(-1)

        return {
            'loss': loss,
            'prediction': prediction,
        }

    def mc(self, ex):
        for k, v in ex["encoding"].items():
            ex["encoding"][k] = v.cuda()

        sent_outputs = self.sent_model(
            **ex["encoding"],
            return_dict=True,
        )
        logits = self.classifier(sent_outputs.last_hidden_state[:, 0, :]).t().view(-1)
        j = 0
        losses = []
        prediction = []
        for i, index in enumerate(ex["index"]):
            losses.append(self.criterion(logits[j:j + index].unsqueeze(dim=0), ex["labels"][i].unsqueeze(dim=0).cuda()))
            prediction.append(torch.argmax(logits[j:j + index]).unsqueeze(dim=0))
            j += index

        loss = torch.mean(torch.cat(losses))
        prediction = torch.cat(prediction).view(-1)
        return {
            "loss": loss,
            "prediction": prediction,
        }

    def resize_token_embeddings(self, length):
        self.cs_model.resize_token_embeddings(length)
        self.sent_model.resize_token_embeddings(length)
          

class GLUEModel(nn.Module):
    def __init__(self, config):
        super(GLUEModel, self).__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config["model"])
        if self.config['task'] == 'mnli':
            self.model_config.num_labels = 3
        elif self.config['task'] == 'sts-b':
            self.model_config.num_labels = 1
        else:
            self.model_config.num_labels = 2

        self.model = AutoModelForSequenceClassification.from_pretrained(self.config["model"], config=self.model_config)

        if self.config['task'] == 'sts-b':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, ex):
        logits = None
        outputs = self.model(
            input_ids=ex["input_ids"].cuda(),
            attention_mask=ex["attention_mask"].cuda(),
            token_type_ids=ex["token_type_ids"].cuda(),
            labels=ex["labels"].cuda(),
        )

        if self.config["task"] == "sts-b":
            prediction = outputs.logits.squeeze()
        else:
            prediction = torch.argmax(outputs.logits, dim=1)
        return {
            'loss': outputs.loss,
            'prediction': prediction,
        }

    def resize_token_embeddings(self, length):
        self.cs_model.resize_token_embeddings(length)
        self.sent_model.resize_token_embeddings(length)


