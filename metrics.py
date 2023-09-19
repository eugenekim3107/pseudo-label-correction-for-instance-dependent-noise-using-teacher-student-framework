import numpy as np
import torch

def SiameseAccuracy(anc, pos, neg):
    with torch.no_grad():
        pos_dist = torch.sqrt(torch.sum((anc - pos)**2,dim=1))
        neg_dist = torch.sqrt(torch.sum((anc - neg)**2,dim=1))
        accuracy = sum(pos_dist < neg_dist) / pos_dist.shape[0]
    return accuracy

def RegularAccuracy(y_pred, y_true):
    with torch.no_grad():
        out = torch.argmax(y_pred, dim=1)
        accuracy = (y_true == out).sum()
        list_len = y_true.shape[0]
    accuracy = accuracy / list_len
    return accuracy