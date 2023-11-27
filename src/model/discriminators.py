import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, period=2, kernel_size=5, stride=3, neg_slope=0.1):
        super(PeriodDiscriminator, self).__init__()
        self.period = period
        self.model = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=((kernel_size - 1) // 2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=128,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=((kernel_size - 1) // 2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=512,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=((kernel_size - 1) // 2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=((kernel_size - 1) // 2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=1024,
                        out_channels=1024,
                        kernel_size=(kernel_size, 1),
                        stride=1,
                        padding=((kernel_size - 1) // 2, 0),
                    )
                ),
            ]
        )
        self.post = weight_norm(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
            )
        )
        self.leakyrelu = nn.LeakyReLU(neg_slope)

    def forward(self, x):
        B, C, T = x.shape
        x = F.pad(
            x,
            (0, int(self.period - T % self.period)),
            "constant",
            0,
        )
        x = x.view(B, C, -1, self.period)

        fmap = []
        for conv in self.model:
            x = conv(x)
            x = self.leakyrelu(x)
            fmap.append(x)

        x = self.post(x)
        fmap.append(x)
        x = torch.flatten(x, 1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                PeriodDiscriminator(period=2),
                PeriodDiscriminator(period=3),
                PeriodDiscriminator(period=5),
                PeriodDiscriminator(period=7),
                PeriodDiscriminator(period=11),
            ]
        )

    def forward(self, prediction, target):
        real_outputs = []
        gen_outputs = []
        real_fmaps = []
        gen_fmaps = []
        for discriminator in self.discriminators:
            real_output, real_fmap = discriminator(target)
            gen_output, gen_fmap = discriminator(prediction)
            real_outputs.append(real_output)
            gen_outputs.append(gen_output)
            real_fmaps.extend(real_fmap)
            gen_fmaps.extend(gen_fmap)

        return real_outputs, gen_outputs, real_fmaps, gen_fmaps


class ScaleDiscriminator(nn.Module):
    def __init__(self, norm=False, neg_slope=0.1):
        super(ScaleDiscriminator, self).__init__()
        blocks = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=128,
                    stride=1,
                    kernel_size=15,
                    padding="same",
                ),
                nn.Conv1d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=41,
                    stride=2,
                    padding=20,
                    groups=4,
                ),
                nn.Conv1d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=41,
                    stride=2,
                    padding=20,
                    groups=16,
                ),
                nn.Conv1d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=16,
                ),
                nn.Conv1d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=16,
                ),
                nn.Conv1d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=41,
                    stride=1,
                    padding=20,
                    groups=16,
                ),
                nn.Conv1d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                ),
            ]
        )
        post = nn.Conv1d(
            in_channels=1024,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if norm:
            blocks = [weight_norm(block) for block in blocks]
            post = weight_norm(post)

        self.model = nn.ModuleList(blocks)
        self.post = post
        self.leakyrelu = nn.LeakyReLU(neg_slope)

    def forward(self, x):
        fmap = []
        for conv in self.model:
            x = conv(x)
            x = self.leakyrelu(x)
            fmap.append(x)

        x = self.post(x)
        fmap.append(x)
        x = torch.flatten(x, 1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(norm=True),
                ScaleDiscriminator(),
                ScaleDiscriminator(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, prediction, target):
        real_outputs = []
        gen_outputs = []
        real_fmaps = []
        gen_fmaps = []
        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                target = self.meanpools[i - 1](target)
                prediction = self.meanpools[i - 1](prediction)
            real_output, real_fmap = discriminator(target)
            gen_output, gen_fmap = discriminator(prediction)
            real_outputs.append(real_output)
            gen_outputs.append(gen_output)
            real_fmaps.extend(real_fmap)
            gen_fmaps.extend(gen_fmap)

        return real_outputs, gen_outputs, real_fmaps, gen_fmaps


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, generated, real):
        real = F.pad(real, (0, generated.shape[-1] - real.shape[-1]), "constant", 0)

        mpd_real_outputs, mpd_gen_outputs, mpd_real_fmaps, mpd_gen_fmaps = self.mpd(
            generated, real
        )
        msd_real_outputs, msd_gen_outputs, msd_real_fmaps, msd_gen_fmaps = self.msd(
            generated, real
        )
        return (
            mpd_real_outputs,
            mpd_gen_outputs,
            mpd_real_fmaps,
            mpd_gen_fmaps,
            msd_real_outputs,
            msd_gen_outputs,
            msd_real_fmaps,
            msd_gen_fmaps,
        )
