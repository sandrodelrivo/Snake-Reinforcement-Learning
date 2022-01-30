import torch.nn as nn
import copy


class ActorNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ActorNet, self).__init__()

        hidden_layer_size = int(input_dim * (2 / 3))

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_size, out_features=output_dim),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.model(x)


class CriticNet(nn.Module):

    def __init__(self, input_dim):
        super(CriticNet, self).__init__()

        hidden_layer_size = int(input_dim * (2 / 3))

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_size, out_features=1),
        )

    def forward(self, x):
        return self.model(x)
