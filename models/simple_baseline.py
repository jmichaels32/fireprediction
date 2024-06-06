import torch
import argparse
import torch.nn as nn
import torchvision.models as models

from utils.model_utils import train, test
from utils.utils import generate_dataloader

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
class simple_baseline(nn.Module):
    def __init__(self):
        super(simple_baseline, self).__init__()

        # Data processing (upscale then downscale)
        self.upscale = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.downscale = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 64 * 64),
            Reshape(64, 64),
        )
    
    def forward(self, x):
        x_iter = x
        x_iter = self.upscale(x_iter)
        x_iter = self.downscale(x_iter)
        x_iter = self.linear(x_iter)
        return x_iter
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    model = simple_baseline()
    train_losses, val_losses = train(model, 'simple_baseline', args.lr, args.batch_size, args.epochs)
    test(model, args.batch_size)