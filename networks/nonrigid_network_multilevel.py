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
            nn.Conv3d(2, 32, 3, stride=2, padding=1),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv3d(34, 32, 3, stride=1, padding=1),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01)
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv3d(66, 64, 3, stride=1, padding=1),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01)
        ) 

        self.encoder_4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
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
            nn.Conv3d(256, 512, 3, stride=2, padding=1),
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

        self.layer_2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01),
            nn.Conv3d(32, 3, 3, stride=1, padding=1),
        )

        self.layer_3 = nn.Sequential(
            nn.Conv3d(64, 32, 3, stride=1, padding=1),
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

    def crop(self, image, template):
        return image[:, :, :template.size(2), :template.size(3), : template.size(4)]
        
    def forward(self, sources, targets):
        sl1, tl1 = sources[0], targets[0]
        sl2, tl2 = sources[1], targets[1]
        sl3, tl3 = sources[2], targets[2]

        x = tc.cat((sl3, tl3), dim=1)
        x1 = self.encoder_1(x)
        x1 = self.crop(x1, sl2)
        x2 = self.encoder_2(tc.cat((sl2, tl2, x1), dim=1))
        x2 = self.crop(x2, sl1)
        x3 = self.encoder_3(tc.cat((sl1, tl1, x2), dim=1))
        x4 = self.encoder_4(x3)
        x5 = self.encoder_5(x4)
        d5 = self.decoder_5(x5)
        d5 = self.crop(d5, x4)
        d4 = self.decoder_4(tc.cat((d5, x4), dim=1))
        d4 = self.crop(d4, x3)
        d3 = self.decoder_3(tc.cat((d4, x3), dim=1))
        d3 = self.crop(d3, x2)
        d2 = self.decoder_2(tc.cat((d3, x2), dim=1))
        d2 = self.pad(d2, x1)
        d1 = self.decoder_1(tc.cat((d2, x1), dim=1))
        d1 = self.pad(d1, x)
        output = [self.layer_3(d3).permute(0, 2, 3, 4, 1), self.layer_2(d2).permute(0, 2, 3, 4, 1), self.layer_1(d1).permute(0, 2, 3, 4, 1)]
        return output

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

    batch_size = 1
    example_sources = [tc.rand((batch_size, no_channels, y_size, x_size, z_size)).to(device), tc.rand((batch_size, no_channels, y_size // 2, x_size // 2, z_size // 2)).to(device), tc.rand((batch_size, no_channels, y_size // 4, x_size // 4, z_size // 4)).to(device)]
    example_targets = [tc.rand((batch_size, no_channels, y_size, x_size, z_size)).to(device), tc.rand((batch_size, no_channels, y_size // 2, x_size // 2, z_size // 2)).to(device), tc.rand((batch_size, no_channels, y_size // 4, x_size // 4, z_size // 4)).to(device)]

    result = model(example_sources, example_targets)
    for item in result:
        print(item.size())


def run():
    test_forward_pass_simple()

if __name__ == "__main__":
    run()
