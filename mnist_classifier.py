# mnist_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.p = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.p(x)
        x = F.relu(self.c2(x))
        x = self.p(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
