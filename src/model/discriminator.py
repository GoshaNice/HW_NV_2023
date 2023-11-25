import torch.nn as nn
import numpy as np


class PeriodDiscriminator(nn.Module):
    def __init__(self, period = 2, kernel_size=5, stride=3):
        self.model = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, 
                out_channels=32, 
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(kernel_size - 1) // 2),
            nn.Conv2d(
                in_channels=32, 
                out_channels=128, 
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(kernel_size - 1) // 2),
            nn.Conv2d(
                in_channels=128, 
                out_channels=512, 
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(kernel_size - 1) // 2),
            nn.Conv2d(
                in_channels=512, 
                out_channels=1024, 
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(kernel_size - 1) // 2),
            nn.Conv2d(
                in_channels=1024, 
                out_channels=1024, 
                kernel_size=(kernel_size, 1),
                stride=1,
                padding=(kernel_size - 1) // 2),
        ])