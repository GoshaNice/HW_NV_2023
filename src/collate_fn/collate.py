import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}

    spectrograms = []
    result_batch["spectrogram_length"] = torch.tensor(
        [item["spectrogram"].shape[-1] for item in dataset_items]
    )
    max_spec_dim_last = torch.max(result_batch["spectrogram_length"])
    for item in dataset_items:
        spectrogram = item["spectrogram"]
        spectrograms.append(
            F.pad(
                spectrogram,
                (0, max_spec_dim_last - spectrogram.shape[-1]),
                "constant",
                0,
            )
        )

    result_batch["spectrogram"] = torch.cat(spectrograms, dim=0)

    audios = []
    result_batch["audio_length"] = torch.tensor(
        [item["audio"].shape[-1] for item in dataset_items]
    )
    max_audio_dim_last = torch.max(result_batch["audio_length"])
    for item in dataset_items:
        audio = item["audio"]
        audios.append(
            F.pad(
                audio,
                (0, max_audio_dim_last - audio.shape[-1]),
                "constant",
                0,
            )
        )

    result_batch["audio"] = torch.cat(audios, dim=0)
    return result_batch
