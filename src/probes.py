import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_regression

def extract_features(model, loader):
    with torch.no_grad():
        pairs = []
        for images, labels in loader:
            y = labels
            if hasattr(model, "encode"):
                x = model.encode(images)
            else: 
                x = model.encoder(images)[:,0]
            pairs.append((x,y))
    x_f, y_f = zip(*pairs)         
    return torch.cat(x_f), torch.cat(y_f)

def fit_probe(x_train,y_train,weight_decay,steps,lr):
    x_mean = x_train.mean(dim=0)
    x_sd = x_train.std(dim=0) + 1.0e-7 # make sure x_sd is never 0
    x_s = (x_train - x_mean)/x_sd
    model = nn.Linear(x_s.shape[1],10)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter = steps)
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_s),y_train) + (model.weight**2).sum()* weight_decay
        loss.backward()
        return loss
    prev = float("inf")
    loss = optimizer.step(closure).item()
    for _ in range(50):
        if abs(loss-prev) > 1.0e-3:
            prev = loss
            loss = optimizer.step(closure).item()
        else: break
    return model, x_mean, x_sd

def probe_accuracy(probe, mean, std, x, y):
    return ((probe((x-mean)/std)).argmax(dim=1)==y).float().mean().item()

import numpy as np

