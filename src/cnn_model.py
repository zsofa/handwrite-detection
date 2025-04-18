import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.convolutional_layer1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pooling_layer = nn.MaxPool2d(2, 2)
        self.convolutional_layer2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fully_connected_layer1 = nn.Linear(64 * 7 * 7, 128)
        self.fully_connected_layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pooling_layer(f.relu(self.convolutional_layer1(x)))
        x = self.pooling_layer(f.relu(self.convolutional_layer2(x)))

        x = x.view(-1, 64 * 7 * 7)

        x = f.relu(self.fully_connected_layer1(x))
        x = self.fully_connected_layer2(x)

        return x