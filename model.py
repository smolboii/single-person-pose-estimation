import torch
from torch import nn
import torchvision.models as models


class PoseEstimationModel(nn.Module):
    def __init__(self, bn_momentum=0.1):
        super(PoseEstimationModel, self).__init__()

        resnet = models.resnet18(pretrained=True)
        resnet_modules = list(resnet.children())[:-2]  # discard avgpool and fc layers
        self.resnet_backbone = nn.Sequential(*resnet_modules)
        
        layers = []
        for _ in range(3):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels = 512, 
                    out_channels = 512,
                    kernel_size = 4,
                    stride = 2,
                    padding = 1,
                    output_padding = 0,
                    bias = False
                ),
            )
            layers.append(nn.BatchNorm2d(512, momentum=bn_momentum))
            layers.append(nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(*layers)

        self.heatmap_out = nn.Sequential(
            nn.Conv2d(
                in_channels = 512,
                out_channels = 16,  # number of joints
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.Sigmoid()  # to constrain between 0 and 1
        )

    def forward(self, x):
        x = self.resnet_backbone(x)
        x = self.deconv(x)
        x = self.heatmap_out(x)

        return x 