import torch
import math
from torch import nn
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput
from transformers.models.roberta.modeling_roberta import RobertaIntermediate, RobertaOutput


class KnowledgeAttention(nn.Module):
    def __init__(self, config):
        super(KnowledgeAttention, self).__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.output = KnowledgeOutput(config)
        self.dropout = nn.Dropout(p=self.config.attention_probs_dropout_prob)

        self.intermediate = BertIntermediate(config) if config.model_type == 'bert' else RobertaIntermediate(config)
        self.output2 = BertOutput(config) if config.model_type == 'bert' else RobertaOutput(config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(1, 0, 2).contiguous()
        # return x.permute(0, 2, 1, 3)
        # batch = x.size()[0]
        # return x.view(self.config.num_attention_heads, batch, self.attention_head_size)

    def forward(self, pooler_output, knowledge, knowledge_mask):
        # print(pooler_output.size(), knowledge.size())
        batch = pooler_output.size()[0]

        mixed_query_layer = self.query(pooler_output)
        mixed_key_layer = self.key(knowledge)
        mixed_value_layer = self.value(knowledge)    

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        if knowledge_mask is not None:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size) + knowledge_mask
        else:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # print(attention_scores[0])
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # print(attention_probs)
        # print(attention_probs)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # print(attention_probs.size())
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(1, 0, 2).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(batch, self.config.hidden_size)

        attention_output = self.output(context_layer, pooler_output)

        output = self.intermediate(attention_output)
        attention_output = self.output2(output, attention_output)

        return attention_output


class KnowledgeOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
