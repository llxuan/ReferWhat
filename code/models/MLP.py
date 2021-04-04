from torch import nn

# the baseline model for the REINFORCE algorithm in reinforcement learning

class MLP2(nn.Module):
    def __init__(self, model_cfg):
        super(MLP2, self).__init__()
        self.FC1 = nn.Linear(model_cfg.IN_SIZE, model_cfg.HIDDEN_SIZE)
        self.activation = nn.ReLU()
        self.FC2 = nn.Linear(model_cfg.HIDDEN_SIZE, 1)

    def forward(self, input):
        hidden = self.activation(self.FC1(input))
        output = self.activation(self.FC2(hidden))
        return output