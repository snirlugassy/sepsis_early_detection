import os
import gc
import csv
import argparse
import pickle

import torch
from torch.utils.data import DataLoader
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
    argparser.add_argument('--data-path', type=str, required=True, dest='data_path', default='/home/student/data/')
    argparser.add_argument('--epochs', type=int, dest='epochs', default=100)
    argparser.add_argument('--optimizer', type=str, dest='optimizer', choices=OPTIMIZERS.keys(), default='adam')
    argparser.add_argument('--lr', type=float, dest='lr', default=0.005)
    argparser.add_argument('--print-steps', type=int, dest='print_steps')

    args = argparser.parse_args()
    
    print('====== TRAIN =======')
    print('optimizer:', args.optimizer)
    print('epochs:', args.epochs)
    print('l-rate:', args.lr)
    print('device:', device)
    print('====================')

    model = SepsisPredictionModel(input_size=len(ICUSepsisDataset.features))
    model.to(device)

    icu_train = ICUSepsisDataset(os.path.join(args.data_path, 'train'))
    train_loader = DataLoader(icu_train, batch_size=1, shuffle=True)
    train_size = len(icu_train)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = OPTIMIZERS[args.optimizer](model.parameters(), lr=args.lr)
 
    for epoch in range(args.epochs):
        train_loss = 0.0
        i = 0
        for x,y in icu_train:
            # ignore invalid samples
            if x is None or y is None:
                continue

            i += 1
            optimizer.zero_grad()

            x = x.to(device).unsqueeze(0)
            y = y.to(device)  # squeeze since batch_size=1
            N = len(y)

            # Forward pass
            output = model(x).view(N,2) # squeeze since batch_size=1
            L = loss(output, y)
            train_loss += L.item() * x.size(0)

            # Backpropagation
            L.backward()
            optimizer.step()

            if i % args.print_steps == 0:
                print(f'L: {train_loss / i: .5}')

        train_loss /= train_size

        print(f'Epoch {epoch+1}/{args.epochs}, Loss {train_loss}')
        
        print('-> Saving state')
        torch.save(model.state_dict(), 'model.state')
