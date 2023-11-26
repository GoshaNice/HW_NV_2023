import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from src.preprocessing.melspectrogram import MelSpectrogram, MelSpectrogramConfig


class GeneratorLoss(nn.Module):
    def __init__(self, lambda_fm, lambda_mel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.wav2spec = MelSpectrogram(MelSpectrogramConfig())
        self.mae = nn.L1Loss()

    def forward(self, spectrogram, prediction, mpd, msd, **batch):
        prediction = prediction.squeeze(1)
        prediction_spectrogram = self.wav2spec(prediction)
        prediction_spectrogram = F.pad(
            prediction_spectrogram,
            (0, int(spectrogram.shape[-1] - prediction_spectrogram.shape[-1])),
            "constant",
            0,
        )
        mel_loss = self.mae(prediction_spectrogram, spectrogram)
        fm_loss = 0
        for gen, real in zip(mpd["gen_fmaps"], mpd["real_fmaps"]):
            fm_loss += self.mae(gen, real)
        for gen, real in zip(msd["gen_fmaps"], msd["real_fmaps"]):
            fm_loss += self.mae(gen, real)

        adversarial_loss = 0
        for prob in mpd["gen_outputs"]:
            adversarial_loss += torch.mean((prob - 1) ** 2)
        for prob in msd["gen_outputs"]:
            adversarial_loss += torch.mean((prob - 1) ** 2)

        generator_loss = (
            adversarial_loss + self.lambda_mel * mel_loss + self.lambda_fm * fm_loss
        )
        return generator_loss, adversarial_loss, mel_loss, fm_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, mpd, msd, **batch):
        discriminator_loss = 0
        for gen, real in zip(mpd["gen_outputs"], mpd["real_outputs"]):
            discriminator_loss += torch.mean((real - 1) ** 2) + torch.mean(gen**2)
        for gen, real in zip(msd["gen_outputs"], msd["real_outputs"]):
            discriminator_loss += torch.mean((real - 1) ** 2) + torch.mean(gen**2)

        return discriminator_loss
