import torch
import torch.nn as nn
import numpy as np
from src.model.generator import Generator
from src.model.discriminators import Discriminator


class HifiGan(nn.Module):
    def __init__(self, model_config):
        super(HifiGan, self).__init__()
        self.generator = Generator(**model_config["generator"])
        self.discriminator = Discriminator()

    def forward(self, spectrogram, **batch):
        output = self.generator(spectrogram)
        return output

    def discriminate(self, prediction, audio, **batch):
        audio = audio.unsqueeze(1)
        discriminator_output = self.discriminator(prediction, audio)
        return discriminator_output
