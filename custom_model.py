# custom_model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (
    partition_input,
    prepare_input_embed_aux_interaction
)


# Modified DSTN model
class CustomModel(nn.Module):
    def __init__(self, num_aux_type, n_one_hot_slot, n_mul_hot_slot, max_num_aux_inst_used, k, total_embed_dim, n_ft,
                 att_hidden_dim=32, temperature=1.5, n_mul_hot_slot_aux=None):
        super(CustomModel, self).__init__()

        # Assign attributes
        self.k = k
        self.num_aux_type = num_aux_type
        self.n_one_hot_slot = n_one_hot_slot
        self.n_mul_hot_slot = n_mul_hot_slot
        self.max_num_aux_inst_used = max_num_aux_inst_used
        self.n_mul_hot_slot_aux = n_mul_hot_slot_aux if n_mul_hot_slot_aux is not None else [n_mul_hot_slot] * num_aux_type
        self.att_hidden_dim = att_hidden_dim
        self.emb_mat = nn.Embedding(n_ft + 1, k)
        self.temperature = temperature

        # Projection layers to align auxiliary embeddings with total_embed_dim
        self.projection_layers = nn.ModuleDict({
            str(i): nn.Linear(total_embed_dim[i], att_hidden_dim) for i in range(num_aux_type)
        })

        # Attention mechanism layers for each auxiliary ad type
        self.attention_layers = nn.ModuleDict({
            str(i): nn.Linear(att_hidden_dim, att_hidden_dim) for i in range(num_aux_type)
        })
        self.att_score_layers = nn.ModuleDict({
            str(i): nn.Linear(att_hidden_dim, 1, bias=False) for i in range(num_aux_type)
        })

        # Normalization layer
        self.norm_layer = nn.LayerNorm(att_hidden_dim)

        # Prediction layers with adjusted input size to match att_hidden_dim
        self.prediction_layers = nn.ModuleDict({
            'context': nn.Sequential(
                nn.Linear(att_hidden_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            ),
            'clicked': nn.Sequential(
                nn.Linear(att_hidden_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            ),
            'unclicked': nn.Sequential(
                nn.Linear(att_hidden_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            )
        })

    # Apply attention mechanism to auxiliary embeddings
    def apply_attention(self, aux_embed, target_embed, aux_type):
        batch_size, max_num_aux_inst_used, _ = aux_embed.shape
        aux_embed = aux_embed.view(batch_size * max_num_aux_inst_used, -1)
        aux_embed = self.projection_layers[str(aux_type)](aux_embed)
        aux_embed = aux_embed.view(batch_size, max_num_aux_inst_used, -1)

        target_embed = self.projection_layers[str(aux_type)](target_embed)
        mask = (aux_embed.abs().sum(dim=-1) > 0).float()

        aux_proj = self.attention_layers[str(aux_type)](aux_embed)
        target_proj = self.attention_layers[str(aux_type)](target_embed)
        target_proj = target_proj.unsqueeze(1).expand(-1, aux_proj.size(1), -1)

        attention_scores = (aux_proj * target_proj).sum(dim=-1)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = F.softmax(attention_scores / self.temperature, dim=-1)

        weighted_aux_embed = (attention_scores.unsqueeze(-1) * aux_proj).sum(dim=1)
        return self.norm_layer(weighted_aux_embed)

    # Retrieve embeddings while masking invalid inputs
    def get_masked_embedding(self, x_input, emb_mat, embedding_dim):
        x_input = torch.clamp(x_input, 0, emb_mat.num_embeddings - 1).long()
        embedded = emb_mat(x_input)
        mask = (x_input > 0).float().unsqueeze(-1)

        if x_input.dim() == 2:
            mask = mask.expand(-1, -1, embedding_dim)
        elif x_input.dim() == 3:
            mask = mask.expand(-1, -1, -1, embedding_dim)

        return embedded * mask

    # Combine embeddings for one-hot and multi-hot features
    def prepare_input_embed(self, x_input_one_hot, x_input_mul_hot):
        data_embed_one_hot = self.get_masked_embedding(x_input_one_hot, self.emb_mat, self.k)
        data_embed_one_hot = data_embed_one_hot.view(-1, self.n_one_hot_slot * self.k)

        data_embed_mul_hot = self.get_masked_embedding(x_input_mul_hot, self.emb_mat, self.k)
        data_embed_mul_hot = data_embed_mul_hot.sum(dim=2).view(-1, self.n_mul_hot_slot * self.k)

        return torch.cat([data_embed_one_hot, data_embed_mul_hot], dim=1)

    # Forward pass of the model
    def forward(self, inputs):
        x_input_one_hot, x_input_mul_hot, x_input_one_hot_aux, x_input_mul_hot_aux = partition_input(inputs)
        target_embed = self.prepare_input_embed(x_input_one_hot, x_input_mul_hot)

        # Apply attention to each auxiliary type
        attention_outputs = []
        for i in range(self.num_aux_type):
            aux_embed = prepare_input_embed_aux_interaction(
                x_input_one_hot_aux[i],
                x_input_mul_hot_aux[i],
                self.max_num_aux_inst_used[i],
                self.n_one_hot_slot,
                self.n_mul_hot_slot_aux[i],
                self.emb_mat
            )
            attention_output = self.apply_attention(aux_embed, target_embed, i)
            attention_outputs.append(attention_output)

        y_ctxt = self.prediction_layers['context'](attention_outputs[0]) if attention_outputs[0].sum().item() != 0 else torch.zeros(
            1).to(target_embed.device)
        y_clicked = self.prediction_layers['clicked'](attention_outputs[1]) if attention_outputs[1].sum().item() != 0 else torch.zeros(
            1).to(target_embed.device)
        y_unclicked = self.prediction_layers['unclicked'](attention_outputs[2]) if attention_outputs[2].sum().item() != 0 else torch.zeros(
            1).to(target_embed.device)

        predictions = [y_ctxt, y_clicked, y_unclicked]
        valid_predictions = [pred for pred in predictions if pred.sum().item() != 0]
        y_avg = torch.mean(torch.stack(valid_predictions), dim=0) if valid_predictions else torch.zeros_like(y_ctxt)

        return y_ctxt, y_clicked, y_unclicked, y_avg
