import numpy as np
import matplotlib.pyplot as plt
import math

import torch as tc
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class Nonrigid_Network(nn.Module):
    def __init__(self, device):
        super(Nonrigid_Network, self).__init__()
        self.device = device

        self.encoder_1 = nn.Sequential(
            nn.Conv3d(2, 32, 3, stride=2, padding=0),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01),
            nn.Conv3d(32, 64, 3, stride=2, padding=0),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01)
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01),
            nn.Conv3d(64, 128, 3, stride=2, padding=0),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01)
        ) 

        self.encoder_4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01),
            nn.Conv3d(128, 256, 3, stride=2, padding=0),
            nn.GroupNorm(256, 256),
            nn.LeakyReLU(0.01)
        )

        self.encoder_5 = nn.Sequential(
            nn.Conv3d(256, 256, 3, stride=1, padding=1),
            nn.GroupNorm(256, 256),
            nn.LeakyReLU(0.01),
            nn.Conv3d(256, 256, 3, stride=1, padding=1),
            nn.GroupNorm(256, 256),
            nn.LeakyReLU(0.01),
            nn.Conv3d(256, 512, 3, stride=2, padding=0),
            nn.GroupNorm(512, 512),
            nn.LeakyReLU(0.01)
        )

        self.decoder_5 = nn.Sequential(
            nn.Conv3d(512, 512, 3, stride=1, padding=1),
            nn.GroupNorm(512, 512),
            nn.LeakyReLU(0.01),
            nn.Conv3d(512, 512, 3, stride=1, padding=1),
            nn.GroupNorm(512, 512),
            nn.LeakyReLU(0.01),
            nn.Conv3d(512, 256, 3, stride=1, padding=1),
            nn.GroupNorm(256, 256),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        )

        self.decoder_4 = nn.Sequential(
            nn.Conv3d(512, 512, 3, stride=1, padding=1),
            nn.GroupNorm(512, 512),
            nn.LeakyReLU(0.01),
            nn.Conv3d(512, 512, 3, stride=1, padding=1),
            nn.GroupNorm(512, 512),
            nn.LeakyReLU(0.01),
            nn.Conv3d(512, 128, 3, stride=1, padding=1),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        )

        self.decoder_3 = nn.Sequential(
            nn.Conv3d(256, 256, 3, stride=1, padding=1),
            nn.GroupNorm(256, 256),
            nn.LeakyReLU(0.01),
            nn.Conv3d(256, 64, 3, stride=1, padding=1),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01),
            nn.Conv3d(128, 32, 3, stride=1, padding=1),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        )

        self.decoder_1 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01),
            nn.Conv3d(64, 32, 3, stride=1, padding=1),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        )

        self.layer_1 = nn.Sequential(
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01),
            nn.Conv3d(32, 3, 3, stride=1, padding=1),
        )

    def pad(self, image, template):
        pad_x = math.fabs(image.size(3) - template.size(3))
        pad_y = math.fabs(image.size(2) - template.size(2))
        pad_z = math.fabs(image.size(4) - template.size(4))
        b_x, e_x = math.floor(pad_x / 2), math.ceil(pad_x / 2)
        b_y, e_y = math.floor(pad_y / 2), math.ceil(pad_y / 2)
        b_z, e_z = math.floor(pad_z / 2), math.ceil(pad_z / 2)
        image = F.pad(image, (b_z, e_z, b_x, e_x, b_y, e_y))
        return image
        
    def forward(self, source, target):
        x = tc.cat((source, target), dim=1)
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)
        x5 = self.encoder_5(x4)
        d5 = self.decoder_5(x5)
        d5 = self.pad(d5, x4)
        d4 = self.decoder_4(tc.cat((d5, x4), dim=1))
        d4 = self.pad(d4, x3)
        d3 = self.decoder_3(tc.cat((d4, x3), dim=1))
        d3 = self.pad(d3, x2)
        d2 = self.decoder_2(tc.cat((d3, x2), dim=1))
        d2 = self.pad(d2, x1)
        d1 = self.decoder_1(tc.cat((d2, x1), dim=1))
        x = self.pad(d1, x)
        x = self.layer_1(x)
        x = x.permute(0, 2, 3, 4, 1)
        return x

def load_network(device, path=None):
    model = Nonrigid_Network(device)
    model = model.to(device)
    if path is not None:
        model.load_state_dict(tc.load(path))
        model.eval()
    return model

def test_forward_pass_simple():
    device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
    model = load_network(device)
    y_size = 128
    x_size = 128
    z_size = 128
    no_channels = 1
    summary(model, [(no_channels, y_size, x_size, z_size), (no_channels, y_size, x_size, z_size)])

    batch_size = 16
    example_source = tc.rand((batch_size, no_channels, y_size, x_size, z_size)).to(device)
    example_target = tc.rand((batch_size, no_channels, y_size, x_size, z_size)).to(device)

    result = model(example_source, example_target)
    print(result.size())


def run():
    test_forward_pass_simple()

if __name__ == "__main__":
    run()
