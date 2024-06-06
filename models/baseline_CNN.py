import os 
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchvision.models as models

from utils.model_utils import train, test

class upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(upsample_block, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(x)
        return x

class conv_net(nn.Module):
    def __init__(self):
        super(conv_net, self).__init__()
        self.encoder = models.mobilenet_v2(pretrained=False).features
        
        self.encoder[0][0] = nn.Conv2d(12, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.layer_names = [
            '0',    # 32x32
            '3',    # 16x16
            '6',    # 8x8
            '13',   # 4x4
            '18',   # 2x2
        ]
        
        self.upsample1 = upsample_block(1280, 512, 3)
        self.upsample2 = upsample_block(608, 256, 3)
        self.upsample3 = upsample_block(288, 128, 3)
        self.upsample4 = upsample_block(152, 64, 3)
        self.upsample5 = upsample_block(96, 32, 3)
        
        self.final_conv = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.output_conv = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x_iter = x
        # Encoding
        features = []
        for name, layer in self.encoder._modules.items():
            x_iter = layer(x_iter)
            if name in self.layer_names:
                features.append(x_iter)
                
        # Decoding
        x_iter = features[-1]
        features = features[:-1][::-1]

        x_iter = self.upsample1(x_iter)
        x_iter = torch.cat([x_iter, features[0]], dim=1)

        x_iter = self.upsample2(x_iter)
        x_iter = torch.cat([x_iter, features[1]], dim=1)

        x_iter = self.upsample3(x_iter)
        x_iter = torch.cat([x_iter, features[2]], dim=1)

        x_iter = self.upsample4(x_iter)
        x_iter = torch.cat([x_iter, features[3]], dim=1)
        
        x_iter = self.upsample5(x_iter)

        x_iter = self.final_conv(x_iter)
        x_iter = self.output_conv(x_iter)
        x_iter = x_iter.squeeze()
        
        return x_iter
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    model = conv_net()
    train_losses, val_losses = train(model, 'baseline', args.lr, args.batch_size, args.epochs)
    test(model, args.batch_size)
    
