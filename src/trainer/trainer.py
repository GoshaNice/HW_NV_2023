import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torchaudio

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker

import numpy as np
import time


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        generator_criterion,
        discriminator_criterion,
        metrics,
        generator_optimizer,
        discriminator_optimizer,
        config,
        device,
        dataloaders,
        generator_lr_scheduler=None,
        discriminator_lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(
            model,
            generator_criterion,
            discriminator_criterion,
            metrics,
            generator_optimizer,
            discriminator_optimizer,
            config,
            device,
        )
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.generator_lr_scheduler = generator_lr_scheduler
        self.discriminator_lr_scheduler = discriminator_lr_scheduler
        self.log_step = 50

        test_melspec = np.load("test_data_folder/mel_1.npy")
        self.test_melspec = torch.from_numpy(test_melspec).to(device)

        self.train_metrics = MetricTracker(
            "discriminator_loss",
            "generator_loss",
            "adversarial_loss",
            "mel_loss",
            "fm_loss",
            "generator grad norm",
            "discriminator grad norm",
            *[m.name for m in self.metrics],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "audio"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update(
                "generator grad norm", self.get_grad_norm_generator()
            )
            self.train_metrics.update(
                "discriminator grad norm", self.get_grad_norm_discriminator()
            )
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Generator Loss: {:.6f}, Discriminator Loss: {:.6f}".format(
                        epoch,
                        self._progress(batch_idx),
                        batch["generator_loss"].item(),
                        batch["discriminator_loss"].item(),
                    )
                )
                self.writer.add_scalar(
                    "generator learning rate",
                    self.generator_lr_scheduler.get_last_lr()[0],
                )
                self.writer.add_scalar(
                    "discriminator learning rate",
                    self.discriminator_lr_scheduler.get_last_lr()[0],
                )
                self._log_spectrogram(batch["spectrogram"])
                self._log_audio(batch["prediction"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics
        if self.generator_lr_scheduler is not None:
            self.generator_lr_scheduler.step()
        if self.discriminator_lr_scheduler is not None:
            self.discriminator_lr_scheduler.step()

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(**batch)
        batch.update(outputs)

        outputs = self.model.discriminate(
            prediction=batch["prediction"].detach(), audio=batch["audio"]
        )
        batch.update(outputs)

        if is_train:
            self.discriminator_optimizer.zero_grad()
            discriminator_loss = self.discriminator_criterion(**batch)
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            self.generator_optimizer.zero_grad()
            outputs = self.model.discriminate(**batch)
            batch.update(outputs)
            (
                generator_loss,
                adversarial_loss,
                mel_loss,
                fm_loss,
            ) = self.generator_criterion(**batch)
            generator_loss.backward()
            self.generator_optimizer.step()
            batch["generator_loss"] = generator_loss
            batch["discriminator_loss"] = discriminator_loss

            metrics.update("generator_loss", generator_loss.item())
            metrics.update("discriminator_loss", discriminator_loss.item())
            metrics.update("adversarial_loss", adversarial_loss.item())
            metrics.update("mel_loss", mel_loss.item())
            metrics.update("fm_loss", fm_loss.item())
            for met in self.metrics:
                metrics.update(met.name, met(**batch))

        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_spectrogram(batch["spectrogram"])
            self._log_audio(batch["prediction"])
            self._log_test_audio()

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    def _log_audio(self, audio_batch):
        audio = random.choice(audio_batch.cpu().detach())
        self.writer.add_audio("audio", audio, sample_rate=22050)

    def _log_test_audio(self):
        self.model.eval()
        output = self.model(self.test_melspec)
        audio = output["prediction"][0].cpu().detach()
        self.writer.add_audio("test_audio", audio, sample_rate=22050)

    @torch.no_grad()
    def get_grad_norm_generator(self, norm_type=2):
        parameters = self.model.generator.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    @torch.no_grad()
    def get_grad_norm_discriminator(self, norm_type=2):
        parameters = self.model.discriminator.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
