
class SnakeNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(SnakeNet, self).__init__()

        '''self.online = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=output_dim),
        )'''

        hidden_layer_size = int(input_dim * (3 / 2))

        self.actor = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_size, out_features=output_dim),
            nn.Softmax()
        )

        self.critic = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_size, out_features=output_dim),
        )

    def forward(self, x, model):

        # print("X:", x)

        if model == "actor":
            return self.actor(x)
        elif model == "critic":
            return self.critic(x)
