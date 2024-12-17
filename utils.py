# utils.py


import torch
import numpy as np
import os
import random
import config_ddpg as cfg


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Confirm gpu usage
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Configurations
k = cfg.k
n_one_hot_slot = cfg.n_one_hot_slot
n_mul_hot_slot = cfg.n_mul_hot_slot
num_aux_type = cfg.num_aux_type
n_one_hot_slot_aux = cfg.n_one_hot_slot_aux
n_mul_hot_slot_aux = cfg.n_mul_hot_slot_aux
max_len_per_slot_aux = cfg.max_len_per_slot_aux
num_aux_inst_in_data = cfg.num_aux_inst_in_data
max_num_aux_inst_used = cfg.max_num_aux_inst_used
max_len_per_slot = cfg.max_len_per_slot


# Set GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Set random seed
torch.manual_seed(12313)
np.random.seed(12313)
random.seed(12313)


# Calculate reward
def calculate_reward(y_pred, Y_bar_t, labels):
    # Calculate a raw reward based on the distance from threshold
    reward = torch.where(
        ((y_pred > Y_bar_t) & (labels == 1)) | ((y_pred < Y_bar_t) & (labels == 0)),
        1 - torch.abs(y_pred - Y_bar_t),  # Positive reward, scaled by proximity to Y_bar_t
        -torch.abs(y_pred - Y_bar_t)      # Negative reward, scaled by distance from Y_bar_t
    )
    # Optionally, clip the reward to ensure it remains within -1 and 1
    clipped_reward = torch.clamp(reward, -1, 1)
    return clipped_reward


# Partition input tensor into one-hot and multi-hot features for target and auxiliary ads
def partition_input(x_input):
    # Ensure x_input is on the GPU and has a batch dimension
    if x_input.dim() == 1:
        x_input = x_input.unsqueeze(0).to(device)  # Move to GPU if needed

    len_list = [n_one_hot_slot, n_mul_hot_slot * max_len_per_slot]
    for i in range(num_aux_type):
        len_list.append(n_one_hot_slot_aux[i] * num_aux_inst_in_data[i])
        len_list.append(n_mul_hot_slot_aux[i] * max_len_per_slot_aux[i] * num_aux_inst_in_data[i])

    idx_list = torch.tensor(np.cumsum(len_list)).to(device)

    x_input_one_hot = x_input[:, :idx_list[0]]
    x_input_mul_hot = x_input[:, idx_list[0]:idx_list[1]].reshape(-1, n_mul_hot_slot, max_len_per_slot)
    x_input_one_hot = torch.nan_to_num(x_input_one_hot, nan=0).to(device)
    x_input_mul_hot = torch.nan_to_num(x_input_mul_hot, nan=0).to(device)

    x_input_one_hot_aux, x_input_mul_hot_aux = {}, {}
    for i in range(num_aux_type):
        temp_1 = x_input[:, idx_list[2 * i + 1]:idx_list[2 * i + 2]].reshape(
            -1, num_aux_inst_in_data[i], n_one_hot_slot_aux[i]
        ).to(device)
        x_input_one_hot_aux[i] = temp_1[:, :max_num_aux_inst_used[i], :]

        temp_2 = x_input[:, idx_list[2 * i + 2]:idx_list[2 * i + 3]].reshape(
            -1, num_aux_inst_in_data[i], n_mul_hot_slot_aux[i], max_len_per_slot_aux[i]
        ).to(device)
        x_input_mul_hot_aux[i] = temp_2[:, :max_num_aux_inst_used[i], :, :]

    return x_input_one_hot, x_input_mul_hot, x_input_one_hot_aux, x_input_mul_hot_aux


# Retrieve one-hot embeddings with masking for valid indices
def get_masked_one_hot(x_input_one_hot, emb_mat):
    # Ensure x_input_one_hot is within embedding matrix range and on the GPU
    x_input_one_hot = torch.clamp(x_input_one_hot, 0, emb_mat.num_embeddings - 1).to(device)
    data_mask = (x_input_one_hot > 0).float().to(device)
    data_embed_one_hot = emb_mat(x_input_one_hot.long()).to(device)  # Embedding should be on GPU
    return data_embed_one_hot * data_mask.unsqueeze(2)


# Retrieve multi-hot embeddings with masking for valid indices
def get_masked_mul_hot(x_input_mul_hot, emb_mat):
    x_input_mul_hot = torch.clamp(x_input_mul_hot.to(device), 0, emb_mat.num_embeddings - 1)
    data_mask = (x_input_mul_hot > 0).float().to(device)
    data_embed_mul_hot = emb_mat(x_input_mul_hot.long())
    return data_embed_mul_hot * data_mask.unsqueeze(3)


# Retrieve one-hot embeddings for auxiliary input with masking
def get_masked_one_hot_aux(x_input_one_hot_ctxt, emb_mat):
    x_input_one_hot_ctxt = x_input_one_hot_ctxt.to(device)
    if torch.isnan(x_input_one_hot_ctxt).any():
        x_input_one_hot_ctxt = torch.nan_to_num(x_input_one_hot_ctxt, nan=0)
    x_input_one_hot_ctxt = torch.clamp(x_input_one_hot_ctxt, 0, emb_mat.num_embeddings - 1)
    data_mask = (x_input_one_hot_ctxt > 0).float().to(device)
    data_embed_one_hot = emb_mat(x_input_one_hot_ctxt.long())
    return data_embed_one_hot * data_mask.unsqueeze(3)


# Retrieve multi-hot embeddings for auxiliary input with masking
def get_masked_mul_hot_aux(x_input_mul_hot_ctxt, emb_mat):
    x_input_mul_hot_ctxt = x_input_mul_hot_ctxt.to(device)
    if torch.isnan(x_input_mul_hot_ctxt).any():
        x_input_mul_hot_ctxt = torch.nan_to_num(x_input_mul_hot_ctxt, nan=0)
    x_input_mul_hot_ctxt = torch.clamp(x_input_mul_hot_ctxt, 0, emb_mat.num_embeddings - 1)
    data_mask = (x_input_mul_hot_ctxt > 0).float().to(device)
    data_embed_mul_hot = emb_mat(x_input_mul_hot_ctxt.long())
    return data_embed_mul_hot * data_mask.unsqueeze(4)


# Prepare combined embeddings for auxiliary ads
def prepare_input_embed(x_input_one_hot, x_input_mul_hot, emb_mat):
    data_embed_one_hot = get_masked_one_hot(x_input_one_hot.to(device), emb_mat).reshape(-1, n_one_hot_slot * k)
    data_embed_mul_hot_pooling = get_masked_mul_hot(x_input_mul_hot.to(device), emb_mat).sum(2).reshape(-1,
                                                                                                        n_mul_hot_slot * k)
    return torch.cat([data_embed_one_hot, data_embed_mul_hot_pooling], axis=1)


def prepare_input_embed_aux_interaction(x_input_one_hot_ctxt, x_input_mul_hot_ctxt, max_num_ctxt, cur_n_one_hot_slot,
                                        cur_n_mul_hot_slot, emb_mat):
    # Get one-hot embeddings
    data_embed_one_hot_ctxt = get_masked_one_hot_aux(x_input_one_hot_ctxt.to(device), emb_mat).reshape(
        -1, max_num_ctxt, cur_n_one_hot_slot * k)

    # Get multi-hot embeddings and pool them
    data_embed_mul_hot_pooling_ctxt = get_masked_mul_hot_aux(x_input_mul_hot_ctxt.to(device), emb_mat).sum(3).reshape(
        -1, max_num_ctxt, cur_n_mul_hot_slot * k)

    # Concatenate one-hot and multi-hot embeddings
    data_embed_ctxt = torch.cat([data_embed_one_hot_ctxt, data_embed_mul_hot_pooling_ctxt], axis=2)

    # Add an explicit check for the expected shape
    expected_dim = cur_n_one_hot_slot * k + cur_n_mul_hot_slot * k
    assert data_embed_ctxt.shape[2] == expected_dim, f"Expected {expected_dim}, got {data_embed_ctxt.shape[2]}"

    return data_embed_ctxt


# Save model checkpoint including optimizer state
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Checkpoint saved at {path}")


# Calculate the average discounted return for a single episode
def calculate_return(rewards, gamma=0.99):
    G = 0  # Initialize return
    for t in range(len(rewards)):
        G += (gamma ** t) * rewards[t]

    # Calculate the average return
    average_return = G / len(rewards) if len(rewards) > 0 else 0
    return average_return
