# trials.py


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import random
import ctr_funcs as func
import config_ddpg as cfg
import shutil
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ttest_ind
from ddpg_agent import DDPGAgent
from dpg_agent import DPGAgent
from custom_model import CustomModel
from utils import (
    calculate_reward,
    save_checkpoint,
    calculate_return
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Configurations
str_txt = cfg.output_file_name
model_saving_addr = './trials_10/'
num_csv_col = cfg.num_csv_col
train_file_name = cfg.train_file_name
val_file_name = cfg.val_file_name
test_file_name = cfg.test_file_name
batch_size = cfg.batch_size
n_ft = cfg.n_ft
k = cfg.k
eta = cfg.eta
kp_prob = cfg.kp_prob
n_epoch = cfg.n_epoch
max_num_lower_ct = cfg.max_num_lower_ct
record_step_size = cfg.record_step_size
layer_dim = cfg.layer_dim
opt_alg = cfg.opt_alg
n_one_hot_slot = cfg.n_one_hot_slot
n_mul_hot_slot = cfg.n_mul_hot_slot
num_aux_type = cfg.num_aux_type
n_one_hot_slot_aux = cfg.n_one_hot_slot_aux
n_mul_hot_slot_aux = cfg.n_mul_hot_slot_aux
max_len_per_slot_aux = cfg.max_len_per_slot_aux
num_aux_inst_in_data = cfg.num_aux_inst_in_data
max_num_aux_inst_used = cfg.max_num_aux_inst_used
max_len_per_slot = cfg.max_len_per_slot
att_hidden_dim = cfg.att_hidden_dim
n_episodes = cfg.n_episodes

# Label initialization
label_col_idx = 0
record_defaults = [[0]] * num_csv_col
record_defaults[0] = [0.0]
total_num_ft_col = num_csv_col - 1

# Create and manage directories
if not os.path.exists(model_saving_addr):
    os.mkdir(model_saving_addr)

if os.path.isdir(model_saving_addr):
    shutil.rmtree(model_saving_addr)
os.mkdir(model_saving_addr)

# Set GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Setup for multiple trials
n_trials = 10  # Number of trials
seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]  # List of predefined seeds

# Storage for rewards and results across trials
ddpg_rewards_all_trials = []
dpg_rewards_all_trials = []
random_rewards_all_trials = []

print('Loading data start!')

# Data input pipeline
def pytorch_input_pipeline(file_names, num_epochs, batch_size, perform_shuffle=True, label_col_idx=0):
    if isinstance(file_names, list):
        data = pd.concat([pd.read_csv(file) for file in file_names], ignore_index=True)
    else:
        data = pd.read_csv(file_names)

    # Separate features and labels
    labels = data.iloc[:, label_col_idx].values
    features = data.drop(columns=data.columns[label_col_idx]).values

    # Convert to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)

    # Create a TensorDataset
    dataset = TensorDataset(features, labels)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=perform_shuffle, num_workers=0)

    return dataloader

# Data augmentation
class AugmentedDataset:
    def __init__(self, dataset, clicked_ratio=3):
        self.dataset = dataset
        self.clicked_ratio = clicked_ratio

        # Separate indices by class
        self.indices_by_class = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self.indices_by_class[int(label.item())].append(idx)

        # Balance dataset
        self.balanced_indices = self._generate_balanced_indices()

    def _generate_balanced_indices(self):
        # Get indices for the underrepresented class
        minority_class = 1
        majority_class = 0

        minority_indices = self.indices_by_class[minority_class]
        majority_indices = self.indices_by_class[majority_class]

        # Oversample minority class
        oversampled_minority_indices = random.choices(minority_indices, k=len(majority_indices) * self.clicked_ratio)

        # Combine and shuffle
        balanced_indices = majority_indices + oversampled_minority_indices
        random.shuffle(balanced_indices)

        return balanced_indices

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, index):
        # Get the balanced index
        balanced_index = self.balanced_indices[index]
        # Retrieve data from the original dataset
        x, y = self.dataset[balanced_index]
        x, y = x.to(device), y.to(device)  # Move to GPU
        return x, y


# Load datasets
train_dataset = func.pytorch_input_pipeline(train_file_name, n_epoch, batch_size, perform_shuffle=True,
                                            label_col_idx=label_col_idx).dataset
val_dataset = func.pytorch_input_pipeline(val_file_name, n_epoch, batch_size, perform_shuffle=False,
                                          label_col_idx=label_col_idx).dataset
test_dataset = func.pytorch_input_pipeline(test_file_name, 1, batch_size, perform_shuffle=False,
                                           label_col_idx=label_col_idx).dataset

# Compute min and max from training data
train_min, train_max = func.compute_min_max(train_dataset)

# Apply min-max normalization
train_dataset_normalized = func.min_max_normalize_dataset(train_dataset, train_min, train_max)
val_dataset_normalized = func.min_max_normalize_dataset(val_dataset, train_min, train_max)
test_dataset_normalized = func.min_max_normalize_dataset(test_dataset, train_min, train_max)

# Wrap datasets with AugmentedDataset
train_dataset_augmented = AugmentedDataset(train_dataset_normalized, clicked_ratio=5)  # You can adjust clicked_ratio
val_dataset_augmented = AugmentedDataset(val_dataset_normalized, clicked_ratio=5)
test_dataset_augmented = AugmentedDataset(test_dataset_normalized, clicked_ratio=5)

# Create DataLoaders
train_loader = DataLoader(train_dataset_augmented, batch_size=batch_size, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset_augmented, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset_augmented, batch_size=batch_size, shuffle=False, num_workers=0)

print('Loading data done!')

for trial in range(n_trials):
    print(f"\nStarting Trial {trial + 1}/{n_trials} with random seed: {seeds[trial]}")

    # Set random seed
    torch.manual_seed(seeds[trial])
    np.random.seed(seeds[trial])
    random.seed(seeds[trial])
    torch.cuda.manual_seed_all(seeds[trial])

    # Calculate total_embed_dim
    total_embed_dim = {i: 250 for i in range(num_aux_type)}

    # Define model
    model = CustomModel(num_aux_type, n_one_hot_slot, n_mul_hot_slot, max_num_aux_inst_used, k, total_embed_dim, n_ft,
                        att_hidden_dim=att_hidden_dim).to(device)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Minimizes validation loss
        factor=0.5,  # Reduces learning rate by half when triggered
        patience=5,  # Waits for 5 epochs without improvement before reducing LR
        threshold=1e-4,  # Minimum change to qualify as an improvement
        cooldown=1,  # Waits for 1 epoch before resuming normal operation after LR change
        verbose=True  # Prints a message when LR is updated
    )
    train_loss_list = []
    val_avg_auc_list = []
    epoch_list = []
    best_n_round = 0
    best_val_avg_auc = 0
    lower_ct = 0

    # Initialize the DDPG agent and set device
    ddpg_agent = DDPGAgent(state_dim=3, action_dim=3, actor_lr=1e-7, critic_lr=1e-6, gamma=0.99, tau=0.001, buffer_size=5000, batch_size=128)

    # Initialize the DPG agent and set device
    dpg_agent = DPGAgent(state_dim=3, action_dim=3, actor_lr=1e-5, critic_lr=1e-5, gamma=0.95, tau=0.0001)

    # Random policy
    def random_policy():
        action = np.random.dirichlet([1, 1, 1])
        return action

    # Start training Custom Model
    print('Start training Custom Model')

    best_val_loss = float('inf')

    total_batches = len(train_loader)

    # Configure weight decay in the optimizer
    optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=1e-4)  # Adjust weight decay as needed


    # Define custom loss with penalty (place this above the training loop)
    def custom_loss_with_penalty(preds, targets, penalty_weight=0.1):
        bce_loss = nn.BCEWithLogitsLoss()(preds, targets)
        # Regularization penalty for overconfidence
        penalty = penalty_weight * ((torch.sigmoid(preds) - 0.5) ** 2).mean()
        return bce_loss + penalty

    for epoch in range(n_epoch):
        model.train()
        total_loss = 0
        num_batches = 0
        data_count = 0

        for batch_idx, (train_ft_batch, train_label_batch) in enumerate(train_loader):
            data_count += len(train_ft_batch)
            train_ft_batch, train_label_batch = train_ft_batch.to(device), train_label_batch.to(device)

            # Get predictions from the model
            y_ctxt, y_clicked, y_unclicked, y_avg = model(train_ft_batch)

            # Compute BCE loss without sigmoid
            loss = custom_loss_with_penalty(y_avg.squeeze(), train_label_batch.squeeze())

            # Add a regularization penalty to prevent overconfidence
            penalty = 0.1 * ((torch.sigmoid(y_avg) - 0.5) ** 2).mean()
            loss += penalty

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_ft_batch, val_label_batch in val_loader:
                val_ft_batch = val_ft_batch.to(device)
                val_label_batch = val_label_batch.to(device)

                # Forward pass
                y_ctxt, y_clicked, y_unclicked, y_avg = model(val_ft_batch)
                val_pred_score = (y_ctxt + y_clicked + y_unclicked).mean(dim=1).squeeze()

                # Calculate validation loss without applying sigmoid to y_avg
                loss = custom_loss_with_penalty(val_pred_score, val_label_batch.squeeze())
                val_loss += loss.item()

                # For AUC calculation, apply sigmoid to the predictions
                y_true = val_label_batch.cpu().numpy().flatten()
                y_pred = torch.sigmoid(val_pred_score).cpu().numpy().flatten()
                auc_score = roc_auc_score(y_true, y_pred)

                # Display some predictions and labels for clarity
                break

        avg_val_loss = val_loss / len(val_loader)

        # Step the scheduler with validation loss
        scheduler.step(avg_val_loss)

        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(model_saving_addr, f'custom_model_checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

    # After all epochs, save the final model state
    final_model_path = os.path.join(model_saving_addr, 'custom_model_final.pth')
    torch.save(model.state_dict(), final_model_path)

    # Initialize variables for cumulative rewards and losses
    ddpg_cumulative_reward = 0
    dpg_cumulative_reward = 0
    ddpg_actions_taken = []
    episode_actor_losses_ddpg = []
    episode_critic_losses_ddpg = []
    episode_actor_losses_dpg = []
    episode_critic_losses_dpg = []
    episode_rewards = []
    cumulative_rewards = []
    dpg_episode_rewards = []
    dpg_cumulative_rewards = []
    ddpg_returns = []
    dpg_returns = []

    # Variables to accumulate loss for averaging every 10 episodes
    total_actor_loss_ddpg = 0
    total_critic_loss_ddpg = 0
    total_actor_loss_dpg = 0
    total_critic_loss_dpg = 0

    for episode in range(n_episodes):
        ddpg_agent.noise.reset()

        # Initialize episode-specific metrics
        actor_losses_ddpg = []
        critic_losses_ddpg = []
        actor_losses_dpg = []
        critic_losses_dpg = []
        episode_rewards_return_ddpg = []
        episode_rewards_return_dpg = []
        max_history_size = 10
        episode_reward_ddpg = 0.0
        episode_reward_dpg = 0.0
        data_count = 0

        # Initialize historical states
        ddpg_state_history = []
        dpg_state_history = []

        for batch_idx, (train_ft_batch, train_label_batch) in enumerate(train_loader):
            # Reset noise for NoisyLinear layers in DDPG if applicable
            ddpg_agent.actor_model.apply(lambda module: module.reset_noise() if hasattr(module, 'reset_noise') else None)

            # Move data to device
            data_count += len(train_ft_batch)
            train_ft_batch = train_ft_batch.to(device)
            train_label_batch = train_label_batch.view(-1, 1).float().to(device)

            # Model predictions
            y_ctxt, y_clicked, y_unclicked, y_avg = model(train_ft_batch)

            # Robbins-Monro Update for DDPG state
            if len(ddpg_state_history) > 0:
                historical_average_ddpg = sum(ddpg_state_history) / len(ddpg_state_history)
                updated_ddpg_state = (
                        (1 - 1 / (episode + 1)) * historical_average_ddpg +
                        (1 / (episode + 1)) * torch.tensor([
                    y_ctxt.mean().item(), y_clicked.mean().item(), y_unclicked.mean().item()
                ]).to(device)
                )
            else:
                updated_ddpg_state = torch.tensor([
                    y_ctxt.mean().item(), y_clicked.mean().item(), y_unclicked.mean().item()
                ]).to(device)

            # Add to history
            ddpg_state_history.append(updated_ddpg_state)
            if len(ddpg_state_history) > max_history_size:  # Optional limit
                ddpg_state_history.pop(0)

            # Robbins-Monro Update for DPG state
            if len(dpg_state_history) > 0:
                historical_average_dpg = sum(dpg_state_history) / len(dpg_state_history)
                updated_dpg_state = (
                        (1 - 1 / (episode + 1)) * historical_average_dpg +
                        (1 / (episode + 1)) * torch.tensor([
                    y_ctxt.mean().item(), y_clicked.mean().item(), y_unclicked.mean().item()
                ]).to(device)
                )
            else:
                updated_dpg_state = torch.tensor([
                    y_ctxt.mean().item(), y_clicked.mean().item(), y_unclicked.mean().item()
                ]).to(device)

            # Add to DPG history
            dpg_state_history.append(updated_dpg_state)
            if len(dpg_state_history) > max_history_size:
                dpg_state_history.pop(0)

            # --- DPG Agent ---
            Y_bar_t = (y_ctxt.mean(dim=1) + y_clicked.mean(dim=1) + y_unclicked.mean(dim=1)) / 3
            action_dpg_logits = dpg_agent.actor(updated_dpg_state.unsqueeze(0))
            action_dpg = dpg_agent.clip_softmax(action_dpg_logits.squeeze(), min_val=0.0, max_val=1.0)
            y_hat_dpg = (action_dpg[0] * y_ctxt + action_dpg[1] * y_clicked + action_dpg[2] * y_unclicked).mean(dim=1)
            reward_dpg = calculate_reward(y_hat_dpg, Y_bar_t, train_label_batch[:y_hat_dpg.shape[0]])
            episode_reward_dpg += reward_dpg.sum().item()
            for reward in reward_dpg:
                episode_rewards_return_dpg.extend(reward.cpu().tolist())

            next_dpg_state = torch.tensor([
                y_ctxt.mean().item(), y_clicked.mean().item(), y_unclicked.mean().item()
            ]).to(device)

            actor_loss_dpg, critic_loss_dpg = dpg_agent.update(
                updated_dpg_state,  # Updated DPG state
                action_dpg,  # Action taken
                reward_dpg.sum().item(),  # Reward from environment
                next_dpg_state,  # Next state
                episode,  # Current episode
                n_episodes  # Total number of episodes
            )

            # Store DPG losses
            if actor_loss_dpg is not None and critic_loss_dpg is not None:
                actor_losses_dpg.append(actor_loss_dpg)
                critic_losses_dpg.append(critic_loss_dpg)

            # --- DDPG Agent ---
            action_ddpg_logits = ddpg_agent.actor_model(updated_ddpg_state.unsqueeze(0))
            action_ddpg = ddpg_agent.clip_softmax(action_ddpg_logits.squeeze(), min_val=0.0, max_val=1.0)
            y_hat_ddpg = (action_ddpg[0] * y_ctxt + action_ddpg[1] * y_clicked + action_ddpg[2] * y_unclicked).mean(dim=1)
            reward_ddpg = calculate_reward(y_hat_ddpg, Y_bar_t, train_label_batch[:y_hat_ddpg.shape[0]])
            reward_ddpg = torch.clamp(reward_ddpg, min=-1, max=1)
            episode_reward_ddpg += reward_ddpg.sum().item()
            for reward in reward_ddpg:
                episode_rewards_return_ddpg.extend(reward.cpu().tolist())

            next_ddpg_state = torch.tensor([
                y_ctxt.mean().item(), y_clicked.mean().item(), y_unclicked.mean().item()
            ]).to(device)

            # Pass prediction and t to remember
            ddpg_agent.remember(
                updated_ddpg_state,  # Updated current state
                action_ddpg,  # Action taken
                reward_ddpg,  # Reward obtained
                next_ddpg_state,  # Next state
                y_hat_ddpg,  # Prediction (or any suitable output like y_hat_ddpg)
                episode + 1  # Current time step or episode number
            )
            actor_loss_ddpg, critic_loss_ddpg = ddpg_agent.train()

            if actor_loss_ddpg is not None and critic_loss_ddpg is not None:
                actor_losses_ddpg.append(actor_loss_ddpg)
                critic_losses_ddpg.append(critic_loss_ddpg)

        # Calculate returns at the end of the episode
        ddpg_return = calculate_return(episode_rewards_return_ddpg, gamma=0.99)
        dpg_return = calculate_return(episode_rewards_return_dpg, gamma=0.99)
        ddpg_returns.append(ddpg_return)
        dpg_returns.append(dpg_return)

        # Update cumulative rewards
        dpg_cumulative_reward += episode_reward_dpg
        dpg_episode_rewards.append(episode_reward_dpg)
        dpg_cumulative_rewards.append(dpg_cumulative_reward)

        ddpg_cumulative_reward += episode_reward_ddpg
        episode_rewards.append(episode_reward_ddpg)
        cumulative_rewards.append(ddpg_cumulative_reward)

        # Store loss for all timestep
        episode_actor_losses_ddpg += actor_losses_ddpg
        episode_critic_losses_ddpg += critic_losses_ddpg
        episode_actor_losses_dpg += actor_losses_dpg
        episode_critic_losses_dpg += critic_losses_dpg

        # Soft update of DDPG target networks
        ddpg_agent.soft_update(ddpg_agent.target_actor_model, ddpg_agent.actor_model, tau=0.001)
        ddpg_agent.soft_update(ddpg_agent.target_critic_model, ddpg_agent.critic_model, tau=0.001)

        # Average loss over 10 episodes
        if (episode + 1) % 10 == 0:
            avg_actor_loss_ddpg = total_actor_loss_ddpg / 10
            avg_critic_loss_ddpg = total_critic_loss_ddpg / 10
            avg_actor_loss_dpg = total_actor_loss_dpg / 10
            avg_critic_loss_dpg = total_critic_loss_dpg / 10

            # Reset accumulated loss
            total_actor_loss_ddpg = 0
            total_critic_loss_ddpg = 0
            total_actor_loss_dpg = 0
            total_critic_loss_dpg = 0

        # Save checkpoints
        if (episode + 1) % 10 == 0:
            ddpg_actor_path = os.path.join(model_saving_addr, f'ddpg_actor_checkpoint_episode_{episode + 1}.pth')
            ddpg_critic_path = os.path.join(model_saving_addr, f'ddpg_critic_checkpoint_episode_{episode + 1}.pth')
            save_checkpoint(ddpg_agent.actor_model, ddpg_agent.actor_optimizer, episode, ddpg_actor_path)
            save_checkpoint(ddpg_agent.critic_model, ddpg_agent.critic_optimizer, episode, ddpg_critic_path)


    # Save the DDPG and agents' actor and critic models after training
    ddpg_actor_model_path = os.path.join(model_saving_addr, "ddpg_actor_model.pth")
    ddpg_critic_model_path = os.path.join(model_saving_addr, "ddpg_critic_model.pth")
    dpg_model_path = os.path.join(model_saving_addr, "dpg_model.pth")

    torch.save(ddpg_agent.actor_model.state_dict(), ddpg_actor_model_path)
    torch.save(ddpg_agent.critic_model.state_dict(), ddpg_critic_model_path)
    torch.save(dpg_agent.actor.state_dict(), dpg_model_path)

    custom_model_path = os.path.join(model_saving_addr, 'custom_model_final.pth')
    if not os.listdir(model_saving_addr):
        print("Model weights not found. Skipping model restoration.")
    else:
        model.load_state_dict(torch.load(custom_model_path))

    if os.path.exists(ddpg_actor_model_path):
        ddpg_agent.actor_model.load_state_dict(torch.load(ddpg_actor_model_path, weights_only=True))
        print("DDPG actor network uploaded.")

    if os.path.exists(ddpg_critic_model_path):
        ddpg_agent.critic_model.load_state_dict(torch.load(ddpg_critic_model_path, weights_only=True))
        print("DDPG critic network uploaded.")


    # DSTN model evaluation
    def evaluate_custom_model(model, test_loader, loss_fn, device):
        model.to(device)
        model.eval()
        test_losses = []
        ctxt_losses = []
        clicked_losses = []
        unclicked_losses = []
        test_pred_ctxt = []
        test_pred_clicked = []
        test_pred_unclicked = []
        test_labels = []

        with torch.no_grad():
            for test_ft_batch, test_label_batch in test_loader:
                # Move data to GPU within the loop
                test_ft_batch, test_label_batch = test_ft_batch.to(device), test_label_batch.to(device)

                # Forward pass
                y_ctxt, y_clicked, y_unclicked, y_avg = model(test_ft_batch)

                # Compute loss for average prediction
                avg_loss = loss_fn(y_avg, test_label_batch.view(-1, 1))
                test_losses.append(avg_loss.item())

                # Calculate individual log losses
                loss_ctxt = loss_fn(y_ctxt.mean(dim=1).squeeze(), test_label_batch.squeeze())
                loss_clicked = loss_fn(y_clicked.mean(dim=1).squeeze(), test_label_batch.squeeze())
                loss_unclicked = loss_fn(y_unclicked.mean(dim=1).squeeze(), test_label_batch.squeeze())

                # Append to lists
                ctxt_losses.append(loss_ctxt.item())
                clicked_losses.append(loss_clicked.item())
                unclicked_losses.append(loss_unclicked.item())

                # Store predictions for AUC
                test_pred_ctxt.extend(torch.sigmoid(y_ctxt.mean(dim=1)).cpu().numpy().flatten())
                test_pred_clicked.extend(torch.sigmoid(y_clicked.mean(dim=1)).cpu().numpy().flatten())
                test_pred_unclicked.extend(torch.sigmoid(y_unclicked.mean(dim=1)).cpu().numpy().flatten())
                test_labels.extend(test_label_batch.cpu().numpy().flatten())

        # Calculate AUC
        auc_ctxt = func.cal_auc(test_pred_ctxt, test_labels)[0]
        auc_clicked = func.cal_auc(test_pred_clicked, test_labels)[0]
        auc_unclicked = func.cal_auc(test_pred_unclicked, test_labels)[0]

        return {
            'test_loss': np.mean(test_losses),
            'AUC_ctxt': auc_ctxt,
            'AUC_clicked': auc_clicked,
            'AUC_unclicked': auc_unclicked,
            'LogLoss_ctxt': np.mean(ctxt_losses),
            'LogLoss_clicked': np.mean(clicked_losses),
            'LogLoss_unclicked': np.mean(unclicked_losses)
        }

    # Evaluate custom model independently
    custom_model_metrics = evaluate_custom_model(model, test_loader, nn.BCEWithLogitsLoss(), device)

    # Print the independent evaluation of the custom model
    print(f"Independent Custom Model AUC - Context: {custom_model_metrics['AUC_ctxt']}")
    print(f"Independent Custom Model AUC - Clicked: {custom_model_metrics['AUC_clicked']}")
    print(f"Independent Custom Model AUC - Unclicked: {custom_model_metrics['AUC_unclicked']}")
    print(f"Independent Custom Model LogLoss - Context: {custom_model_metrics['LogLoss_ctxt']}")
    print(f"Independent Custom Model LogLoss - Clicked: {custom_model_metrics['LogLoss_clicked']}")
    print(f"Independent Custom Model LogLoss - Unclicked: {custom_model_metrics['LogLoss_unclicked']}")
    print(f"Custom Model Test Loss (Average): {custom_model_metrics['test_loss']}")
    print("Evaluating Custom Model with DDPG...")

    # Initialize lists for storing results
    ddpg_results = {
        'Trial': [],
        'Mean Reward DDPG': [],
        'Mean Reward DPG': [],
        'Mean Reward Random': [],
        'Test Loss': [],
        'AUC': []
    }

    # Function to calculate AUC and RMSE
    def calculate_auc_rmse(predictions, labels):
        auc = func.cal_auc(predictions, labels)[0] if hasattr(func, 'cal_auc') else 0
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(labels)) ** 2))
        return auc, rmse

    # Move models to device
    model.to(device)
    ddpg_agent.actor_model.to(device)
    ddpg_agent.critic_model.to(device)
    dpg_agent.actor.to(device)
    dpg_agent.critic.to(device)

    # Original test dataset without shuffling, full test set for single evaluation
    original_test_dataset = pytorch_input_pipeline(test_file_name, 1, batch_size, perform_shuffle=False,
                                                   label_col_idx=label_col_idx).dataset
    test_loader = DataLoader(original_test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize lists for metrics
    ddpg_trial_rewards = []
    dpg_trial_rewards = []
    random_trial_rewards = []
    ddpg_trial_weights = []
    dpg_trial_weights = []
    random_trial_weights = []
    test_pred_scores_ddpg = []
    test_pred_scores_dpg = []
    test_pred_scores_random = []
    test_labels = []
    ddpg_rewards = []
    dpg_rewards = []
    random_rewards = []
    state_history_ddpg = []
    state_history_dpg = []

    # Begin evaluation on the entire test set
    with torch.no_grad():
        for test_ft_batch, test_label_batch in test_loader:
            # Move data to the appropriate device
            test_ft_batch = test_ft_batch.to(device)
            test_label_batch = test_label_batch.to(device)

            # Append true labels
            test_labels.extend(test_label_batch.cpu().numpy().flatten())

            # Model predictions
            y_ctxt, y_clicked, y_unclicked, y_avg = model(test_ft_batch)
            y_avg = torch.sigmoid(y_avg)  # Apply sigmoid to y_avg if needed for probability predictions

            # Apply sigmoid to all predictions for consistent probabilities
            y_ctxt, y_clicked, y_unclicked = map(torch.sigmoid, [y_ctxt, y_clicked, y_unclicked])

            # Calculate Y_bar_t as the mean of non-zero predictions
            available_preds = [y for y in [y_ctxt, y_clicked, y_unclicked] if y.sum().item() != 0]
            Y_bar_t = torch.mean(torch.stack(available_preds), dim=0).squeeze() if available_preds else torch.zeros_like(
                y_ctxt).squeeze()

            # DDPG State Update
            if len(state_history_ddpg) > 0:
                historical_average_ddpg = torch.stack(state_history_ddpg).mean(dim=0)
            else:
                historical_average_ddpg = torch.zeros(3, device=device)

            recent_contribution_ddpg = (1 / (len(state_history_ddpg) + 1)) * torch.tensor([
                y_ctxt.mean().item() if y_ctxt.sum().item() != 0 else 0,
                y_clicked.mean().item() if y_clicked.sum().item() != 0 else 0,
                y_unclicked.mean().item() if y_unclicked.sum().item() != 0 else 0
            ]).to(device)

            state_ddpg = (1 - 1 / (len(state_history_ddpg) + 1)) * historical_average_ddpg + recent_contribution_ddpg
            state_history_ddpg.append(state_ddpg)

            # DDPG Action
            ddpg_action_logits = ddpg_agent.actor_model(state_ddpg.unsqueeze(0))
            ddpg_action = ddpg_agent.clip_softmax(ddpg_action_logits.squeeze(), min_val=0.0, max_val=1.0).cpu().numpy()

            # Weighted combination of predictions based on action
            y_hat_ddpg = sum(
                w * pred for w, pred in zip(ddpg_action, [y_ctxt, y_clicked, y_unclicked]) if pred.sum().item() != 0)
            y_hat_ddpg = y_hat_ddpg.mean(dim=1).squeeze()

            # Reward calculation
            reward_ddpg = calculate_reward(y_hat_ddpg, Y_bar_t, test_label_batch[:y_hat_ddpg.shape[0]])
            average_reward_ddpg = reward_ddpg.sum().item() / (reward_ddpg.shape[0] * reward_ddpg.shape[1])
            ddpg_rewards.append(average_reward_ddpg)

            # Append DDPG predictions for AUC/RMSE calculations
            test_pred_scores_ddpg.extend(y_hat_ddpg.cpu().numpy().flatten())

            # DPG State Update
            if len(state_history_dpg) > 0:
                historical_average_dpg = torch.stack(state_history_dpg).mean(dim=0)
            else:
                historical_average_dpg = torch.zeros(3, device=device)

            recent_contribution_dpg = (1 / (len(state_history_dpg) + 1)) * torch.tensor([
                y_ctxt.mean().item() if y_ctxt.sum().item() != 0 else 0,
                y_clicked.mean().item() if y_clicked.sum().item() != 0 else 0,
                y_unclicked.mean().item() if y_unclicked.sum().item() != 0 else 0
            ]).to(device)

            state_dpg = (1 - 1 / (len(state_history_dpg) + 1)) * historical_average_dpg + recent_contribution_dpg
            state_history_dpg.append(state_dpg)

            # DPG Action
            dpg_action_logits = dpg_agent.actor(state_dpg.unsqueeze(0))
            dpg_action = dpg_agent.clip_softmax(dpg_action_logits.squeeze(), min_val=0.0, max_val=1.0).cpu().numpy()

            # Weighted combination of predictions based on action
            y_hat_dpg = sum(
                w * pred for w, pred in zip(dpg_action, [y_ctxt, y_clicked, y_unclicked]) if pred.sum().item() != 0)
            y_hat_dpg = y_hat_dpg.mean(dim=1).squeeze()

            # Reward calculation
            reward_dpg = calculate_reward(y_hat_dpg, Y_bar_t, test_label_batch[:y_hat_dpg.shape[0]])
            average_reward_dpg = reward_dpg.sum().item() / (reward_dpg.shape[0] * reward_dpg.shape[1])
            dpg_rewards.append(average_reward_dpg)

            # Append DPG predictions for AUC/RMSE calculations
            test_pred_scores_dpg.extend(y_hat_dpg.cpu().numpy().flatten())

            # Random Policy
            random_action = random_policy()
            random_pred_score = sum(w * pred for w, pred in zip(random_action, [y_ctxt, y_clicked, y_unclicked]) if
                                    pred.sum().item() != 0).mean(dim=1).squeeze()
            test_pred_scores_random.extend(random_pred_score.cpu().numpy().flatten())

            # Random Policy reward calculation
            random_reward = calculate_reward(random_pred_score, Y_bar_t, test_label_batch[:random_pred_score.shape[0]])
            average_reward_random = random_reward.sum().item() / (random_reward.shape[0] * random_reward.shape[1])
            random_rewards.append(average_reward_random)

    # Append results for this trial
    ddpg_rewards_all_trials.append(np.mean(ddpg_rewards))
    dpg_rewards_all_trials.append(np.mean(dpg_rewards))
    random_rewards_all_trials.append(np.mean(random_rewards))


# Convert rewards to NumPy arrays for arithmetic operations
ddpg_rewards = np.array(ddpg_rewards_all_trials)
dpg_rewards = np.array(dpg_rewards_all_trials)
random_rewards = np.array(random_rewards_all_trials)

# Perform T-Tests for each comparison
t_stat_ddpg_random, p_val_ddpg_random = ttest_ind(ddpg_rewards, random_rewards, equal_var=False)
t_stat_ddpg_dpg, p_val_ddpg_dpg = ttest_ind(ddpg_rewards, dpg_rewards, equal_var=False)
t_stat_dpg_random, p_val_dpg_random = ttest_ind(dpg_rewards, random_rewards, equal_var=False)

# Display statistical results in a DataFrame, similar to Table 5 format
df = pd.DataFrame({
    "Comparison": ["DDPG vs Random", "DDPG vs DPG", "DPG vs Random"],
    "df": [len(ddpg_rewards) - 1, len(ddpg_rewards) - 1, len(dpg_rewards) - 1],
    "t": [round(t_stat_ddpg_random, 4), round(t_stat_ddpg_dpg, 4), round(t_stat_dpg_random, 4)],
    "p": [round(p_val_ddpg_random, 4), round(p_val_ddpg_dpg, 4), round(p_val_dpg_random, 4)],
    "95% CI": [
        f"{np.percentile(ddpg_rewards - random_rewards, 2.5):.2f}, {np.percentile(ddpg_rewards - random_rewards, 97.5):.2f}",
        f"{np.percentile(ddpg_rewards - dpg_rewards, 2.5):.2f}, {np.percentile(ddpg_rewards - dpg_rewards, 97.5):.2f}",
        f"{np.percentile(dpg_rewards - random_rewards, 2.5):.2f}, {np.percentile(dpg_rewards - random_rewards, 97.5):.2f}"
    ]
})
print(df.to_string(index=False))