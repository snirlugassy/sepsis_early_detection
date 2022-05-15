from unicodedata import bidirectional
import torch

# Model A without dropout (model_a_1)
class SepsisPredictionModel_A1(torch.nn.Module):
    def __init__(self, input_size, hidden_dim=100):
        super(SepsisPredictionModel_A1, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 100), 
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 20), 
            torch.nn.ReLU(), 
            torch.nn.Linear(20, 2)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.mlp(x)
        return x

# Model A with dropout (model_a_2)
class SepsisPredictionModel(torch.nn.Module):
    def __init__(self, input_size, hidden_dim=100):
        super(SepsisPredictionModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 100), 
            torch.nn.ReLU(), 
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 20), 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(20, 2)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.mlp(x)
        return x

class SepsisPredictionModel_B1(torch.nn.Module):
    def __init__(self, input_size, hidden_dim=100):
        super(SepsisPredictionModel_B1, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 100), 
            torch.nn.ReLU(), 
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(100, 20), 
            torch.nn.ReLU(),
            torch.nn.Linear(20, 2)
        )

    def forward(self, x):
        # only from pytorch 1.11 there is support for unbatched LSTM
        # in case there is no batch dimension, add it
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x, _ = self.lstm(x)

        # considers only the last state for predicting sepsis
        x = x.squeeze()[-1].unsqueeze(0)
        
        x = self.mlp(x)
        
        return x

class SepsisPredictionModel_C1(torch.nn.Module):
    def __init__(self, input_size, hidden_dim=128):
        super(SepsisPredictionModel_C1, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size=hidden_dim, batch_first=True, num_layers=3, bidirectional=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_dim, 128), 
            torch.nn.Tanh(), 
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 32), 
            torch.nn.Tanh(),
            torch.nn.Linear(32, 2)
        )

    def forward(self, x):
        # only from pytorch 1.11 there is support for unbatched LSTM
        # in case there is no batch dimension, add it
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x, _ = self.lstm(x)

        # considers only the last state for predicting sepsis
        x = x.squeeze()[-1]

        x = self.mlp(x)
        return x