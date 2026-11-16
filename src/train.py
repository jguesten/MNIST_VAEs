import torch
from collections import defaultdict


def train_one_epoch(model, optimizer, loader):
        total_loss=0
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = model.loss_fn(batch)
            loss.backward()
            optimizer.step()
            ## logging
            total_loss+=loss.item()*len(batch[0])
            print(loss.item())
        return total_loss/len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    totals, n = {}, 0
    with torch.no_grad():
        for batch in loader:
            size = len(batch[0])
            for key, value in model.metrics(batch).items():
                totals[key] = totals.get(key, 0.0) + value * size ##.get(key, 0.0), check if key already exists, otherwise assign 0.0 (happens for first batch)
            n += size
    return {key: total / n for key, total in totals.items()}

# %% fitting model
def fit(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = defaultdict(list)
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader)
        history["train_loss"].append(train_loss)
        for key, value in evaluate(model, val_loader).items():
            history[key].append(value)
    return history