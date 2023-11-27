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
        prediction = self.generator(spectrogram)
        return {"prediction": prediction}

    def discriminate(self, prediction, audio, **batch):
        audio = audio.unsqueeze(1)
        (
            mpd_real_outputs,
            mpd_gen_outputs,
            mpd_real_fmaps,
            mpd_gen_fmaps,
            msd_real_outputs,
            msd_gen_outputs,
            msd_real_fmaps,
            msd_gen_fmaps,
        ) = self.discriminator(prediction, audio)
        return {
            "mpd_real_outputs": mpd_real_outputs,
            "mpd_gen_outputs": mpd_gen_outputs,
            "mpd_real_fmaps": mpd_real_fmaps,
            "mpd_gen_fmaps": mpd_gen_fmaps,
            "msd_real_outputs": msd_real_outputs,
            "msd_gen_outputs": msd_gen_outputs,
            "msd_real_fmaps": msd_real_fmaps,
            "msd_gen_fmaps": msd_gen_fmaps,
        }
