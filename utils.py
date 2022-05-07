import numpy as np


def wsum(v:list) -> float:
    """
    weighted sum of [(w1,x1), (w2,x2),...]
    where wi is the weight of x1
    """
    s = 0
    ws = 0
    if len(v) == 0:
        return 0

    for w,x in v:
        s += w*x
        ws += w
    return s / ws


def impute_window_avg(x, w_size=1, decay=2):
    n = len(x)
    _x = x.copy()
    for i in range(len(x)):
        v = []
        if np.isnan(x[i]):
            k = 0
            while len(v) == 0:
                for j in range(1,w_size+k):
                    if i-j>0 and not np.isnan(x[i-j]):
                        v.append(((1/j)**decay, x[i-j]))
                    if i+j < n and not np.isnan(x[i+j]):
                        v.append(((1/j)**decay, x[i+j]))
                if len(v) > 0:
                    _x[i] = wsum(v)
                k += 1
    return _x


def calc_accuracy(data_loader, model, sample_size=None):
    if sample_size is None:
        sample_size = len(data_loader)
    total = 0
    correct = 0
    i = 0
    for x,y in data_loader:
        i += 1
        _y = model(x)
        correct += sum(_y.argmax(dim=2).flatten() == y.flatten())
        total += len(y.flatten())
        i += 1
        if i >= sample_size:
            break
    
    return correct / total
