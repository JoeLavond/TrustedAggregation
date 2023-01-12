# packages
import torch.nn as nn


# standardize
class StdChannels(nn.Module):

    def __init__(self, mu, sd):
        super(StdChannels, self).__init__()
        self.register_buffer('mu', mu.view(len(mu), 1, 1))
        self.register_buffer('sd', sd.view(len(sd), 1, 1))

    def forward(self, x):
        return (x - self.mu) / self.sd
