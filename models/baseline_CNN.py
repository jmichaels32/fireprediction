import os 
import torch
import argparse
import numpy as np
from torch import nn

from utils.utils import generate_dataloader
from utils.model_utils import loss

'''def conv_net(nn.Module):
    def __init__(self):
        super(conv_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 64*14*14)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
def train(model, train_data, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for i, data in enumerate(train_data):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")
    print('Finished Training')'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default='train')
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train_data = generate_dataloader('eval', args.batch_size)

    total_pixels = 0
    total_nodata = 0
    total_nofire = 0
    total_fire = 0
    for train in train_data:
        seg_mask_prev = train[:, -2, :, :]
        seg_mask_gold = train[:, -1, :, :]

        total_pixels += np.prod(seg_mask_prev.shape) + np.prod(seg_mask_gold.shape)
        total_nodata += np.sum(seg_mask_prev.numpy() == -1) + np.sum(seg_mask_gold.numpy() == -1)
        total_nofire += np.sum(seg_mask_prev.numpy() == 0) + np.sum(seg_mask_gold.numpy() == 0)
        total_fire += np.sum(seg_mask_prev.numpy() == 1) + np.sum(seg_mask_gold.numpy() == 1)
        
    print(f"Total Pixels: {total_pixels}")
    print(f"Total No Data: {total_nodata}")
    print(f"Total No Data: {100 * total_nodata/total_pixels}")
    print(f"Total No Fire: {total_nofire}")
    print(f"Total No Fire: {100 * total_nofire/total_pixels}")
    print(f"Total Fire: {total_fire}")
    print(f"Total Fire: {100 * total_fire/total_pixels}")
    
