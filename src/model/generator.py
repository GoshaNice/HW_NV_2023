import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, channels, kernel, dilations, neg_slope=0.1):
        super(ResBlock, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(dilations)):
            block = nn.Sequential(
                nn.LeakyReLU(negative_slope=neg_slope),
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    padding="same",
                    dilation=dilations[i],
                ),
                nn.LeakyReLU(negative_slope=neg_slope),
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    padding="same",
                    dilation=1,
                ),
            )
            self.blocks.append(block)

    def forward(self, x):
        output = 0
        for block in self.blocks:
            output = output + block(x)
        return output


class MRF(nn.Module):
    def __init__(self, channels, kernels, dilations, neg_slope=0.1):
        super(MRF, self).__init__()
        num_blocks = len(dilations)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                ResBlock(
                    channels=channels,
                    kernel=kernels[i],
                    dilations=dilations[i],
                    neg_slope=neg_slope,
                )
            )

    def forward(self, x):
        output = self.blocks[0](x)
        for i in range(1, self.num_blocks):
            output += self.blocks[i](x)

        return output


class Generator(nn.Module):
    def __init__(
        self,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        initial_channels,
        n_mels,
        pre_kernel_size: int = 7,
        post_kernel_size: int = 7,
        neg_slope: float = 0.1,
    ):
        super(Generator, self).__init__()
        self.pre = nn.Conv1d(
            in_channels=n_mels,
            out_channels=initial_channels,
            kernel_size=pre_kernel_size,
            padding="same",
        )
        self.upsample_blocks = nn.ModuleList()
        self.num_blocks = len(upsample_rates)
        for i in range(self.num_blocks):
            block = nn.Sequential(
                nn.LeakyReLU(neg_slope),
                nn.ConvTranspose1d(
                    in_channels=initial_channels // np.power(2, i),
                    out_channels=initial_channels // np.power(2, i + 1),
                    kernel_size=upsample_kernel_sizes[i],
                    stride=upsample_rates[i],
                    padding=(upsample_kernel_sizes[i] - upsample_rates[i]) // 2,
                ),
                MRF(
                    channels=initial_channels // np.power(2, i + 1),
                    kernels=resblock_kernel_sizes,
                    dilations=resblock_dilation_sizes,
                ),
            )
            self.upsample_blocks.append(block)

        self.post = nn.Sequential(
            nn.LeakyReLU(negative_slope=neg_slope),
            nn.Conv1d(
                in_channels=initial_channels // np.power(2, self.num_blocks),
                out_channels=1,
                kernel_size=post_kernel_size,
                padding="same",
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.pre(x)
        for block in self.upsample_blocks:
            x = block(x)
        x = self.post(x)
        return x
