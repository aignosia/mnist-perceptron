import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy_score(y: torch.Tensor, y_pred: torch.Tensor):
    return torch.eq(y, y_pred).type(torch.float).mean().item() * 100


def process_data(path: str):
    data = np.load(path)
    X_train = torch.from_numpy(data["x_train"]).type(torch.float).to(device)
    y_train = torch.from_numpy(data["y_train"]).type(torch.long).to(device)
    X_test = torch.from_numpy(data["x_test"]).type(torch.float).to(device)
    y_test = torch.from_numpy(data["y_test"]).type(torch.long).to(device)

    return {
        "x_train": X_train.reshape(X_train.shape[0], -1),
        "y_train": y_train,
        "x_test": X_test.reshape(X_test.shape[0], -1),
        "y_test": y_test,
    }


def main():
    print("Processing data...")
    torch.manual_seed(42)

    data = process_data("data/mnist.npz")
    X_train, y_train = data["x_train"], data["y_train"]
    X_test, y_test = data["x_test"], data["y_test"]

    print("Setting model...")
    model = nn.Sequential(nn.Linear(in_features=784, out_features=10)).to(device)

    lr = 1e-2
    batch = 128
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

    results = {"lr": lr, "batch": batch, "results": []}
    epochs = 4
    print("Training model...")
    for epoch in range(epochs):
        X_train_dl = DataLoader(X_train, batch_size=batch)
        y_train_dl = DataLoader(y_train, batch_size=batch)
        for X_batch, ybatch in zip(X_train_dl, y_train_dl):
            model.train()
            y_logits = model(X_train)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            loss = loss_fn(y_logits, y_train)
            acc = accuracy_score(y_train, y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()

            base_path = Path("models/")
        with torch.inference_mode():
            y_test_logits = model(X_test)
            y_test_pred = torch.softmax(y_test_logits, dim=1).argmax(dim=1)
            test_loss = loss_fn(y_test_logits, y_test)
            test_acc = accuracy_score(y_test, y_test_pred)
            print(
                f"Epoch: {epoch + 1}"
                + f"\n  Train loss: {loss:.4f} | Train acc.: {acc:.4f}"
                + f"\n  Test loss:  {test_loss:.4f} | Test acc.:  {test_acc:.4f}"
            )
            res = {
                "epoch": epoch,
                "train_loss": loss.item(),
                "train_acc": acc,
                "test_loss": test_loss.item(),
                "test_acc": test_acc,
            }
            results["results"].append(res)

    path = base_path.joinpath(f"pytorch_{epochs}")
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path.joinpath("model.pkl"))
    with open(path.joinpath("results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
