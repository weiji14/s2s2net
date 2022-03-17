# S2S2Net Python application

This folder contains Python scripts used to pre-process satellite data, as well
as the neural network model architecture, data loaders, and training/testing
scripts. To ensure high standards of reproducibility, the code is structured
using the [Pytorch Lightning](https://www.pytorchlightning.ai) framework and
based on https://github.com/PyTorchLightning/deep-learning-project-template.

- :bricks: data_aligner.py - Code to reproject and align the high spatial resolution mask with Sentinel-2 imagery
- :mahjong: data_chipper.py - Chops up images into smaller chips in a grid fashion with some overlap
- :spider_web: model.py - Code containing Neural Network model architecture and data loading modules
