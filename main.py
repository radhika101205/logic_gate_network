import torch
import torch.nn as nn
import time
from model import LogicGateNet
from data import generate_logic_dataset
from config import LEARNING_RATE, EPOCHS
from utils import accuracy

def train():
    model = LogicGateNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    dataloader = generate_logic_dataset()

    for epoch in range(EPOCHS):
        total_acc = 0
        for X, y in dataloader:
            out = model(X)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_acc += accuracy(out, y)
        print(f"Epoch {epoch+1}, Accuracy: {total_acc / len(dataloader):.4f}")

    # Inference timing
    model.eval()
    X, y = next(iter(dataloader))
    with torch.no_grad():
        start = time.perf_counter()
        out = model(X)
        end = time.perf_counter()
        acc = accuracy(out, y)
        print(f"\n Final Accuracy on sample batch: {acc:.4f}")
        print(f" Inference time for {len(X)} samples: {(end - start):.6f} seconds")
        print(f" Average inference time per sample: {(end - start)/len(X)*1e6:.2f} µs")

if __name__ == "__main__":
    train()
