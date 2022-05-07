import os
import gc
import csv
import argparse
import pickle

import torch
import pandas as pd
import numpy as np
import tqdm
from scipy.interpolate import interp1d

from dataset import ICUSepsisDataset
from model import SepsisPredictionModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adadelta': torch.optim.Adadelta,
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad
}

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train mask detection nerual network')
    argparser.add_argument('--data-path', type=str, required=True, dest='data_path', default='/home/student/data/train')
    argparser.add_argument('--epochs', type=int, dest='epochs', default=100)
    argparser.add_argument('--optimizer', type=str, dest='optimizer', choices=OPTIMIZERS.keys(), default='adam')
    argparser.add_argument('--lr', type=float, dest='lr', default=0.005)
    argparser.add_argument('--print-steps', type=int, dest='print_steps')

    args = argparser.parse_args()
    
    print('====== TRAIN =======')
    print('optimizer:', args.optimizer)
    print('batch-size:', args.batch_size)
    print('epochs:', args.epochs)
    print('l-rate:', args.lr)
    print('device:', device)
    print('====================')

    model = SepsisPredictionModel(input_size=len(ICUSepsisDataset.features))
    model.to(device)

    icu_train = ICUSepsisDataset(args.data_path)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
 
    for epoch in range(args.epochs):
        train_loss = 0.0
        for x,y in icu_train:
            if x is None or y is None:
                continue

            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            # Forward pass
            output = model(x)
            L = loss(output, y)
            train_loss += L.item() * x.size(0)

            # Backpropagation
            L.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{args.epochs}, Loss {train_loss}')
        
        print('-> Saving state')
        torch.save(model.state_dict(), 'model.state')
