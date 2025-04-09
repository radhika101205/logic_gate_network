import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from config import BATCH_SIZE, INPUT_DIM
import random
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def generate_logic_dataset(n_samples=1000, return_raw=False, save_csv=True):

    # Generate binary inputs
    X = np.random.randint(0, 2, size=(n_samples, INPUT_DIM))

    # Unpack inputs
    X1, X2 = X[:, 0], X[:, 1]
    X3, X4 = X[:, 2], X[:, 3]
    X5, X6 = X[:, 4], X[:, 5]
    X7, X8 = X[:, 6], X[:, 7]

    # Compute labels from logic rule
    part1 = (X1 & (~X2)) | (X3 ^ X4)
    part2 = X5 | (~X6)
    part3 = (X7 == X8)
    y = ((part1 & part2) | part3).astype(np.int64)

    # Save to CSV (optional)
    if save_csv:
        df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(INPUT_DIM)])
        df["label"] = y
        df.to_csv("logic_dataset.csv", index=False)
        print("📁 Dataset saved to logic_dataset.csv")

    # Return PyTorch dataset
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    if return_raw:
        return X_tensor, y_tensor
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
