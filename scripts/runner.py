#!/usr/bin/env python3

import tqdm
import cv2
import numpy as np
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchzq
from functools import partial
from pathlib import Path
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.append(".")

from scripts.difussion import Diffusion


class Runner(torchzq.LegacyRunner):
    def __init__(
        self,
        root: Path = "data/processed",
        base_size=[144, 144],
        crop_size=[128, 128],
        ds_repeat: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.update_args(locals(), ["self", "kwargs"])

    def create_dataset(self):
        dataset = self.autofeed(
            ImageFolder,
            dict(
                transform=transforms.Compose(
                    [
                        transforms.Resize(self.args.base_size),
                        transforms.RandomCrop(self.args.crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 1),
                    ]
                )
            ),
        )
        return ConcatDataset([dataset] * self.args.ds_repeat)

    def create_model(self):
        model = Diffusion(
            num_classes=1,
            class_dim=128,
            factors=[2, 2, 2, 2, 2],
            up_pre_dim=384,
            up_post_dim=3,
            up_dims=[256, 256, 128, 64, 64],
            up_dilations=[
                [1, 2, 1, 2],
                [1, 2, 1, 2],
                [1, 2, 4, 8],
                [1, 2, 4, 8],
                [1, 2, 4, 8],
            ],
            down_pre_dim=32,
            down_dims=[64, 64, 128, 256],
            down_dilations=[
                [1, 2, 4],
                [1, 2, 4],
                [1, 2, 4],
                [1, 2, 4],
            ],
        )
        model.set_new_noise_schedule(torch.linspace(1e-4, 0.05, 50))
        return model

    def prepare_batch(self, batch):
        x, y = batch
        return x.to(self.args.device), None

    def feed(self, x):
        return x

    def criterion(self, x, y):
        c = torch.zeros(len(x)).long().to(x.device)
        return self.model.compute_loss(c, x)

    def initialize(self):
        super().initialize()
        args = self.args

        if self.training:

            def plot(iteration):
                if iteration % args.plot_every == 0:
                    generated = self.model.sample(torch.zeros(16).long().cuda())
                    self.logger.add_images(
                        "generated",
                        (generated + 0.5).clamp(0, 1),
                    )
                    self.logger.render(iteration)

            self.events.iteration_completed.append(plot)

    @torchzq.command
    def train(self, plot_every: int = 10, **kwargs):
        self.update_args(locals(), ["self", "kwargs"])
        super().train(**kwargs)

    @torchzq.command
    def preprocess(self, raw_root: Path = "data/raw"):
        args = self.args

        def slice_iter():
            stride = 167
            size = 163
            for path in sorted(raw_root.glob("*.jpg")):
                image = cv2.imread(str(path))
                image = np.pad(image, ((0, 1), (0, 0), (0, 0)))
                for i in range(0, image.shape[0], stride):
                    for j in range(0, image.shape[1], stride):
                        yield image[i : i + stride, j : j + stride][:size, :size]

        outdir = args.root / "0"
        outdir.mkdir(parents=True, exist_ok=True)

        for i, slice in enumerate(tqdm.tqdm(slice_iter())):
            cv2.imwrite(str(Path(outdir, f"{i:06d}.png")), slice)


if __name__ == "__main__":
    torchzq.start(Runner)
