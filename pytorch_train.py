import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def accuracy_score(y: torch.Tensor, y_pred: torch.Tensor):
    return torch.eq(y, y_pred).type(torch.float).mean().item() * 100


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Processing data...")
data = np.load("data/mnist.npz")
X_train = torch.from_numpy(data["x_train"]).type(torch.float).to(device)
X_train = X_train.reshape(X_train.shape[0], -1)
y_train = torch.from_numpy(data["y_train"]).type(torch.long).to(device)
X_test = torch.from_numpy(data["x_test"]).type(torch.float).to(device)
X_test.reshape(X_test.shape[0], -1)
y_test = torch.from_numpy(data["y_test"]).type(torch.long).to(device)
dataset = TensorDataset(X_train, y_train)

print("Setting model...")
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(in_features=784, out_features=10)).to(device)
lr = 1e-2
batch = 128
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
results = {"lr": lr, "batch": batch, "results": []}
epochs = 4

print("Training model...")
for epoch in range(epochs):
    train_dl = DataLoader(dataset, batch_size=batch)
    for X_batch, y_batch in train_dl:
        model.train()
        y_logits = model(X_train)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_score(y_train, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

    with torch.inference_mode():
        y_test_logits = model(X_test)
        y_test_pred = torch.softmax(y_test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(y_test_logits, y_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(
            f"Epoch: {epoch + 1}\n  Loss: {loss:.4f} | Acc.: {acc:.4f}"
            + f"\n  Test loss:  {test_loss:.4f} | Test acc.:  {test_acc:.4f}"
        )
        res = {
            "epoch": epoch + 1,
            "loss": loss.item(),
            "acc": acc,
            "test_loss": test_loss.item(),
            "test_acc": test_acc,
        }
        results["results"].append(res)

print("Saving results...")
base_path = Path("models/")
path = base_path.joinpath("pytorch")
path.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), path.joinpath("model.pkl"))
with open(path.joinpath("results.json"), "w") as f:
    json.dump(results, f, indent=2)
