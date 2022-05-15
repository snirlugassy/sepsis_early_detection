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


def calc_accuracy(predicted:np.array, labels:np.array):
    return sum(predicted == labels) / len(labels)


def calc_f1_score(predicted:np.array, labels:np.array):
    tp = int(np.logical_and(labels==1, predicted==1).sum())
    fp = int(np.logical_and(labels==0, predicted==1).sum())
    fn = int(np.logical_and(labels==1, predicted==0).sum())
    tn = int(np.logical_and(labels==0, predicted==0).sum())
    print('tp = ', tp)
    print('fp = ', fp)
    print('fn = ', fn)
    print('tn = ', tn)
    
    return 2 * tp / (2 * tp + fp + fn)