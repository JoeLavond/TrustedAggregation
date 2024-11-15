# packages
import torch
import torch.nn as nn


# standardize
class StdChannels(nn.Module):
    """
    Standardize input channels
    For use as a first layer in a neural network

    """

    def __init__(self, mu: torch.Tensor, sd: torch.Tensor):
        """
        Initialize standardization layer
        Register mean and standard deviation as buffers
        Buffers are not updated during training

        Args:
            mu (torch.Tensor): mean of each channel
            sd (torch.Tensor): standard deviation of each channel

        """
        super(StdChannels, self).__init__()
        self.register_buffer('mu', mu.view(len(mu), 1, 1))
        self.register_buffer('sd', sd.view(len(sd), 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standardize input tensor

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: standardized tensor

        """
        return (x - self.mu) / self.sd
