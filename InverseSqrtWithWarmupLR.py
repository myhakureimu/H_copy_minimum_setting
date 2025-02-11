import torch
import torch.nn as nn
import torch.optim as optim
import math

class InverseSqrtWithWarmupLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_epochs, base_lr):
        """
        Implements inverse square root decay with warm-up.
        :param optimizer: PyTorch optimizer
        :param warmup_epochs: Number of warm-up epochs
        :param base_lr: Maximum learning rate after warm-up
        """
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs  # Linear warm-up
        else:
            return (self.warmup_epochs ** 0.5) / (epoch ** 0.5)  # Inverse sqrt decay