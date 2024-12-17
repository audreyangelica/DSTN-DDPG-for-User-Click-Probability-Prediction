# ctr_funcs.py


import torch
import numpy as np
from sklearn import metrics
import pandas as pd
from torch.utils.data import DataLoader


# Compute the minimum and maximum values for each feature in the dataset
def compute_min_max(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))
    inputs, _ = data  # Assuming the inputs are the first element, labels are the second
    min_vals = inputs[:, 1:].min(dim=0).values  # Exclude the first column (label)
    max_vals = inputs[:, 1:].max(dim=0).values  # Exclude the first column (label)
    return min_vals, max_vals


# Create a dataset wrapper that applies Min-Max normalization to the features
def min_max_normalize_dataset(dataset, min_vals, max_vals):
    class MinMaxNormalizedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, min_vals, max_vals):
            self.dataset = dataset
            self.min_vals = min_vals
            self.max_vals = max_vals

        # Return the size of the dataset
        def __len__(self):
            return len(self.dataset)

        # Normalize features excluding the label, and add the label back
        def __getitem__(self, idx):
            data, label = self.dataset[idx]
            features = data[1:]  # Exclude the label
            normalized_features = (features - self.min_vals) / (self.max_vals - self.min_vals)
            return torch.cat([data[:1], normalized_features]), label  # Add back the label

    return MinMaxNormalizedDataset(dataset, min_vals, max_vals)


# Calculate the Area Under the ROC Curve (AUC) for predicted scores
def cal_auc(pred_score, label):
    fpr, tpr, thresholds = metrics.roc_curve(label, pred_score, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    return auc_val, fpr, tpr


# Calculate the Root Mean Square Error (RMSE) between predicted scores and true labels
def cal_rmse(pred_score, label):
    mse = metrics.mean_squared_error(label, pred_score)
    rmse = np.sqrt(mse)
    return rmse


# Compute RMSE with rectified predictions using a sample rate adjustment
def cal_rectified_rmse(pred_score, label, sample_rate):
    for idx, item in enumerate(pred_score):
        pred_score[idx] = item / (item + (1 - item) / sample_rate)
    mse = metrics.mean_squared_error(label, pred_score)
    rmse = np.sqrt(mse)
    return rmse


# Flatten a list of lists into a single list
def list_flatten(input_list):
    output_list = [yy for xx in input_list for yy in xx]
    return output_list


# Count the number of lines in a file
def count_lines(file_name):
    num_lines = sum(1 for line in open(file_name, 'rt'))
    return num_lines


# PyTorch DataLoader helper functions
def pytorch_input_pipeline(file_name, num_epochs, batch_size, perform_shuffle=True, label_col_idx=0):
    # Load your data using pandas
    data = pd.read_csv(file_name)

    # Separate features and labels
    labels = data.iloc[:, label_col_idx].values
    features = data.drop(columns=data.columns[label_col_idx]).values

    # Convert to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Add dimension for labels

    # Create a TensorDataset
    dataset = torch.utils.data.TensorDataset(features, labels)

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=perform_shuffle, num_workers=4)

    return dataloader


# Load and preprocess test data from a CSV file into a PyTorch DataLoader
def pytorch_input_pipeline_test(file_name, batch_size, num_epochs=1, label_col_idx=0):
    # Load your data using pandas
    data = pd.read_csv(file_name)

    # Separate features and labels
    labels = data.iloc[:, label_col_idx].values
    features = data.drop(columns=data.columns[label_col_idx]).values

    # Convert to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Add dimension for labels

    # Create a TensorDataset
    dataset = torch.utils.data.TensorDataset(features, labels)

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return dataloader
