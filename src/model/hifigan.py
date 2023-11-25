import torch
import torch.nn as nn
import numpy as np
from src.model.generator import Generator


class HifiGan(nn.Module):
    def __init__(self, model_config):
        super(HifiGan, self).__init__()
        self.generator = Generator(**model_config["generator"])
        self.mpd = None
        self.msd = None

    def forward(self, spectrogram, **batch):
        output = self.generator(spectrogram)
        if self.training:
            logits_mps = self.mpd(output)
            logits_msd = self.msd(output)
            return output, logits_mps, logits_msd
        else:
            return output
