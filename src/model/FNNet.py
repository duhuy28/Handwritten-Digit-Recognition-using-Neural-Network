import torch.nn as nn

class FNNet(nn.Module):
    def __init__(self):
        super(FNNet, self).__init__()
        hidden_1 = 512
        hidden_2 = 200
        self.model = nn.Sequential(nn.Linear(28 * 28, hidden_1),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_1, hidden_2),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_2, 10)
                                    )


    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.model(x)
        return x