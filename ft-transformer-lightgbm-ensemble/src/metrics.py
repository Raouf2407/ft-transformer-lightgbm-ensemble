import numpy as np
import torch

def hit_rate(y_true, y_pred, delta=0.05):
    # metrique du concours : proportion de predictions a moins de delta de la vraie valeur
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred) <= delta)

def hit_rate_lgbm(y_pred, y_true, delta=0.05):
    if hasattr(y_true, "get_label"):
        y_true = y_true.get_label()
    return "hit_rate", hit_rate(y_true, y_pred, delta), True

def focused_loss(pred, target, delta=0.05, reg=0.01):
    err = torch.abs(pred - target)
    depassement = torch.relu(err - delta)
    return torch.mean(depassement ** 2) + reg * torch.mean(err)
