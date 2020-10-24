# Single person pose estimation

## Description

This project is an ANN model for estimating single person poses from RGB image inputs, inspired by the model in the paper [Simple Baselines for Human Pose Estimation
and Tracking](https://arxiv.org/pdf/1804.06208.pdf). The model features a ResNet18 backbone from the torchvision.models Python module (pre-trained), with the last two layers (Average pooling and FC layer) removed. Connected to this backbone is then 3 transposed convolutional (/deconvolutional) layers, which upscale the output feature maps of the ResNet18, with the output of the model being a heatmap for joint-location probabilities, with one channel for each joint (16 total).

### Dataset used

The dataset used for training is the [LIP dataset](http://sysu-hcp.net/lip/overview.php), which is a modification of the typical COCO dataset whereby, for the most part, each image has been cropped to contain only a single person. Hence, if one desires to train this model themselves, it is recommended to *not* use the COCO dataset, as the model does not have the means to predict poses for multiple people. The LIP dataset does not necessarily have to be used, however the dataset used should only contain images with a single person in them.

### Training progress

As of right now the model has not finished training. I am training it on a lone RTX 1660 Ti GPU using CUDA through PyTorch, and have thus far only completed roughly ~20 epochs. Currently, the average MSE loss is at ~0.00029, and is steadily decreasing. I will likely train it for ~100 epochs total, and as such it is not that close to being finished. Checkpoint models will be uploaded here periodically to mark the progress of the model (in the checkpoints/ directory).

## Usage

### CLI
![Command line help text](/res/cli.png)


