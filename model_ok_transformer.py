import torch
from models.modeling_bert import (
    BertWithKnowledgeForMaskedLM,
    BertWithKnowledgeForMultipleChoice,
    BertWithKnowledgeForSequenceClassification,
    BertWithKnowledgeForQuestionAnswering,
)

from models.modeling_roberta import (
    RobertaModelWithKnowledge,
    RobertaWithKnowledgeForMaskedLM,
    RobertaWithKnowledgeForSequenceClassification,
    RobertaWithKnowledgeForMultipleChoice,
)

from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig, 
    AutoModel, 
    BertForMaskedLM, 
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    RobertaForMaskedLM,
    RobertaTokenizer,
    RobertaModel,
)


WINOGRAD_MODEL_CLASS_1 = {
    "bert-base-uncased": BertWithKnowledgeForSequenceClassification,
    "bert-large-uncased": BertWithKnowledgeForSequenceClassification,
    "roberta-base": RobertaWithKnowledgeForSequenceClassification,
    "roberta-large": RobertaWithKnowledgeForSequenceClassification,
}

WINOGRAD_MODEL_CLASS_2 = {
    "bert-base-uncased": BertWithKnowledgeForMaskedLM,
    "bert-large-uncased": BertWithKnowledgeForMaskedLM,
    "roberta-base": RobertaWithKnowledgeForMaskedLM,
    "roberta-large": RobertaWithKnowledgeForMaskedLM,
}

GLUE_MODEL_CLASS = {
    "bert-base-uncased": BertWithKnowledgeForSequenceClassification,
    "bert-large-uncased": BertWithKnowledgeForSequenceClassification,
    "roberta-base": RobertaWithKnowledgeForSequenceClassification,
    "roberta-large": RobertaWithKnowledgeForSequenceClassification,
}

SQUAD_MODEL_CLASS = {
    "bert-base-uncased": BertWithKnowledgeForQuestionAnswering,
    "bert-large-uncased": BertWithKnowledgeForQuestionAnswering,
    # "roberta-base": RobertaWithKnowledgeForQuestionAnswering,
    # "roberta-large": RobertaWithKnowledgeForQuestionAnswering, 
}

MC_MODEL_CLASS = {
    "bert-base-uncased": BertWithKnowledgeForMultipleChoice,
    "bert-large-uncased": BertWithKnowledgeForMultipleChoice,
    "roberta-large": RobertaWithKnowledgeForMultipleChoice,
}

MLM_MODEL_CLASS = {
    "bert-base-uncased": BertWithKnowledgeForMaskedLM,
    "bert-large-uncased": BertWithKnowledgeForMaskedLM, 
}

SC_MODEL_CLASS = {
    "bert-base-uncased": BertWithKnowledgeForSequenceClassification,
    "roberta-large": RobertaWithKnowledgeForSequenceClassification,
}

class WinogradModel(nn.Module):
    def __init__(self, config):
        super(WinogradModel, self).__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config["model"])
        self.model_config.output_hidden_states = True
        self.model_config.gradient_checkpointing = True if self.config["gradient_checkpoint"] else False
        self.model_config.add_pooler_output = True if self.config["add_pooler_output"] else False
        self.model_config.last_pooler_output = True if self.config["last_pooler_output"] else False
        
        self.cs_model = AutoModel.from_pretrained(self.config["model"], config=self.model_config)
        if self.config["method"] == "mask":
            self.sent_model = WINOGRAD_MODEL_CLASS_2[self.config['model']].from_pretrained(self.config["model"], config=self.model_config)
        else:
            self.model_config.num_labels = 1
            self.sent_model = WINOGRAD_MODEL_CLASS_1[self.config['model']].from_pretrained(self.config["model"], config=self.model_config)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, data):
        if self.config["method"] == "mask":
            return self.mask(data)
        else:
            return self.mc(data)
    
    def mask(self, data):
        if not self.config["static"]:
            knowledge = [[] for i in range(self.model_config.num_hidden_layers)]
            with torch.no_grad():
                for commonsense in data["commonsense"]:
                    for k, v in commonsense.items():
                        commonsense[k] = v.cuda()
                    cs_outputs = self.cs_model(
                        **commonsense,
                        return_dict=True,
                        output_hidden_states=True,
                    )

                    batch_knowledge = cs_outputs.hidden_states[1:]
                    batch_knowledge = list(batch_knowledge)
                    for j in range(len(batch_knowledge)):
                        knowledge[j].append(batch_knowledge[j][:, 0, :])
            
            for i, kn in enumerate(knowledge):
                knowledge[i] = torch.cat(kn).requires_grad_() 

        data["knowledge_mask"] = (1.0 - data["knowledge_mask"]) * -100000.0

        for k, v in data["encoding"].items():
            data["encoding"][k] = v.cuda()
        
        sent_outputs = self.sent_model(
            **data["encoding"],
            knowledge=knowledge if not self.config["static"] else data["knowledge"].cuda(),
            knowledge_mask=data["knowledge_mask"].cuda(),
            return_dict=True,
        )

        logits = sent_outputs.logits
        losses = torch.mean(self.criterion(logits.view(-1, self.model_config.vocab_size), data["answers"].view(-1).cuda()).view(logits.size()[:2]), dim=1).squeeze()
        batch_size = len(data["labels"])
        temp = torch.zeros(batch_size, 5).cuda()
        old_index = -1
        j = 0
        
        for i, index in enumerate(data["index"]):
            if index == old_index:
                j += 1
            else:
                j = 0
            temp[index][j] = losses[i]
            old_index = index

        loss = torch.zeros(1).squeeze().cuda()
        # Not suitable for PDP dataset!
        for i in range(batch_size):
            loss += temp[i][data["labels"][i]] + self.config["alpha"] * torch.relu(temp[i][data["labels"][i]] - temp[i][torch.relu(1 - data["labels"][i])] + self.config["beta"])
        # loss = torch.mean(temp[torch.arange(batch_size), data['labels']].squeeze())  # + self.config['alpha'] * torch.sum(nn.functional.relu(temp[torch.arange(batch_size), data['labels']] - temp[torch.arange(batch_size), :1] + self.config['beta']).t()) / batch_size
        prediction = torch.argmin(torch.where(temp == 0, 10000. * torch.ones(temp.size()).long().cuda(), temp), dim=1)
        loss /= batch_size

        return {
            'loss': loss,
            'prediction': prediction,
            'knowledge': knowledge if not self.config["static"] else None,
        }

    def mc(self, data):
        if not self.config["static"]:
            knowledge = [[] for i in range(self.model_config.num_hidden_layers)]
            with torch.no_grad():
                for commonsense in data["commonsense"]:
                    for k, v in commonsense.items():
                        commonsense[k] = v.cuda()
                    cs_outputs = self.cs_model(
                        **commonsense,
                        return_dict=True,
                        output_hidden_states=True,
                    )

                    batch_knowledge = cs_outputs.pooler_output if self.config["last_pooler_output"] else cs_outputs.hidden_states[1:]
                    batch_knowledge = list(batch_knowledge)
                    for j in range(len(batch_knowledge)):
                        knowledge[j].append(batch_knowledge[j][:, 0, :])
            
            for i, kn in enumerate(knowledge):
                knowledge[i] = torch.cat(kn).requires_grad_() 

        data["knowledge_mask"] = (1.0 - data["knowledge_mask"]) * -100000.0

        for k, v in data["encoding"].items():
            data["encoding"][k] = v.cuda()

        sent_outputs = self.sent_model(
            **data["encoding"],
            knowledge=knowledge if not self.config["static"] else data["knowledge"].cuda(),
            knowledge_mask=data["knowledge_mask"].cuda(),
            return_dict=True
        )

        logits = sent_outputs.logits.squeeze()
        
        j = 0
        losses = []
        prediction = []
        for i, index in enumerate(data["index"]):
            losses.append(self.criterion(logits[j:j + index].unsqueeze(dim=0), data["labels"][i].unsqueeze(dim=0).cuda()))
            prediction.append(torch.argmax(logits[j:j + index]).unsqueeze(dim=0))
            j += index

        loss = torch.mean(torch.cat(losses))
        prediction = torch.cat(prediction).view(-1)

        return {
            "loss": loss,
            "prediction": prediction,
            "knowledge": knowledge if not self.config["static"] else None,
        }

    def resize_token_embeddings(self, length):
        # self.cs_model.resize_token_embeddings(length)
        self.sent_model.resize_token_embeddings(length)

    def update_cs(self, data, knowledge_grad):
        for i, commonsense in enumerate(data["commonsense"]):
            for k, v in commonsense.items():
                commonsense[k] = v.cuda()

            cs_outputs = self.cs_model(
                **commonsense,
                return_dict=True,
                output_hidden_states=True,
            )

            new_knowledge = cs_outputs.pooler_output if self.config["last_pooler_output"] else cs_outputs.hidden_states[1:]
            new_knowledge = torch.cat([kn[:, 0, :].unsqueeze(dim=0) for kn in new_knowledge]) # 12 * 128 * 768
            new_knowledge_grad = torch.cat([kn[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]].unsqueeze(dim=0) for kn in knowledge_grad])
            new_knowledge.backward(new_knowledge_grad)
           

class GLUEModel(nn.Module):
    def __init__(self, config):
        super(GLUEModel, self).__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config["model"])
        self.model_config.output_hidden_states = True
        self.model_config.gradient_checkpointing = True if self.config["gradient_checkpoint"] else False
        self.model_config.add_pooler_output = True if self.config["add_pooler_output"] else False
        self.model_config.last_pooler_output = True if self.config["last_pooler_output"] else False

        if self.config['task'] == 'mnli':
            self.model_config.num_labels = 3
        elif self.config['task'] == 'sts-b':
            self.model_config.num_labels = 1
        else:
            self.model_config.num_labels = 2

        self.cs_model = AutoModel.from_pretrained(config["model"], config=self.model_config)
        
        self.sent_model = GLUE_MODEL_CLASS[self.config['model']].from_pretrained(self.config["model"], config=self.model_config)
        if self.config['task'] == 'sts-b':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, data):
        if not self.config["static"]:
            knowledge = [[] for i in range(self.model_config.num_hidden_layers)]
            with torch.no_grad():
                for commonsense in data["commonsense"]:
                    for k, v in commonsense.items():
                        commonsense[k] = v.cuda()
                    cs_outputs = self.cs_model(
                        **commonsense,
                        return_dict=True,
                        output_hidden_states=True,
                    )

                    batch_knowledge = cs_outputs.pooler_output if self.config["last_pooler_output"] else cs_outputs.hidden_states[1:]
                    batch_knowledge = list(batch_knowledge)
                    for j in range(len(batch_knowledge)):
                        knowledge[j].append(batch_knowledge[j][:, 0, :])
            
            for i, kn in enumerate(knowledge):
                knowledge[i] = torch.cat(kn).requires_grad_() 

        data["knowledge_mask"] = (1.0 - data["knowledge_mask"]) * -100000.0

        for k, v in data["encoding1"].items():
            data["encoding1"][k] = v.cuda()

        sent_outputs = self.sent_model(
            **data["encoding1"],
            knowledge=knowledge if not self.config["static"] else data["knowledge"].cuda(),
            knowledge_mask=data["knowledge_mask"].cuda(),
            return_dict=True,
            labels=data["labels"].cuda() if "t5" in self.config["model"] else None,
        )
        
        logits = sent_outputs.logits
        if self.config['task'] != 'sts-b':
            loss = self.criterion(logits, data['labels'].cuda())
            prediction = torch.argmax(logits, dim=1)
        else:
            loss = self.criterion(logits.squeeze(), data["labels"].float().cuda())
            prediction = logits.squeeze()

        return {
            'loss': loss,
            'prediction': prediction,
            'knowledge': knowledge if not self.config["static"] else None,
        }

    def resize_token_embeddings(self, length):
        self.cs_model.resize_token_embeddings(length)
        self.sent_model.resize_token_embeddings(length)
    
    def update_cs(self, data, knowledge_grad):
        for i, commonsense in enumerate(data["commonsense"]):
            for k, v in commonsense.items():
                commonsense[k] = v.cuda()

            cs_outputs = self.cs_model(
                **commonsense,
                return_dict=True,
                output_hidden_states=True,
            )

            new_knowledge = cs_outputs.pooler_output if self.config["last_pooler_output"] else cs_outputs.hidden_states[1:]
            new_knowledge = torch.cat([kn[:, 0, :].unsqueeze(dim=0) for kn in new_knowledge]) # 12 * 128 * 768
            new_knowledge_grad = torch.cat([kn[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]].unsqueeze(dim=0) for kn in knowledge_grad])
            new_knowledge.backward(new_knowledge_grad)

    def encode_cs(self, data):
        knowledge = [[] for i in range(self.model_config.num_hidden_layers)]
        for i, commonsense in enumerate(data["commonsense"]):
            for k, v in commonsense.items():
                commonsense[k] = v.cuda()

            cs_outputs = self.cs_model(
                **commonsense,
                return_dict=True,
                output_hidden_states=True,
            )

            batch_knowledge = cs_outputs.hidden_states[1:]
            batch_knowledge = list(batch_knowledge)
            for j in range(len(batch_knowledge)):
                knowledge[j].append(batch_knowledge[j][:, 0, :])

        for i, kn in enumerate(knowledge):
            knowledge[i] = torch.cat(kn)
        
        return knowledge
    
    def evaluate(self, data, knowledge):
        data["knowledge_mask"] = (1.0 - data["knowledge_mask"]) * -100000.0
        for k, v in data["encoding1"].items():
            data["encoding1"][k] = v.cuda()

        sent_outputs = self.sent_model(
            **data["encoding1"],
            knowledge=knowledge,
            knowledge_mask=data["knowledge_mask"].cuda(),
            return_dict=True,
        )
        logits = sent_outputs.logits
        if self.config['task'] != 'sts-b':
            loss = self.criterion(logits, data['labels'].cuda())
            prediction = torch.argmax(logits, dim=1)
        else:
            loss = self.criterion(logits.squeeze(), data["labels"].float().cuda())
            prediction = logits.squeeze()

        return {
            'loss': loss,
            'prediction': prediction,
            'logits': logits,
        }
