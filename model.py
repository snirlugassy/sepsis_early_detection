import torch


# 0.927 accuracy using model_0_927_acc.state
# class SepsisPredictionModel(torch.nn.Module):
#     def __init__(self, input_size, hidden_dim=100):
#         super(SepsisPredictionModel, self).__init__()
#         self.lstm = torch.nn.LSTM(input_size, hidden_size=hidden_dim, batch_first=True, num_layers=2)
#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(hidden_dim, 100), 
#             torch.nn.ReLU(), 
#             torch.nn.Linear(100, 20), 
#             torch.nn.ReLU(), 
#             torch.nn.Linear(20, 2)
#         )

    
#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = self.mlp(x)
#         return x

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
