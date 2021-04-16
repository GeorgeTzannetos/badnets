from torch import nn
import torch.nn.functional as F


class BadNet(nn.Module):
    """ Badnet model class based on the description of table1 of the paper with two convolution
    and two fully connected layers """
    def __init__(self, input_size=3, output=10):
        super().__init__()
        self.input_size = input_size
        self.output = output
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        if input_size == 3:
            self.fc_features = 800
        else:
            self.fc_features = 512
        self.fc1 = nn.Linear(self.fc_features, 512)
        self.fc2 = nn.Linear(512, output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.fc_features)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x

