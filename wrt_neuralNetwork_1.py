import torch # for all things PyTorch
import torch.nn as nn # To access the parent object for Pytorch's model
from torch.utils.data import DataLoader # To access Pytorch's dataloader
import torch.nn.functional as F # To access Pytorch's activation function

class wrt_neuralNetwork_1(nn.Module):

    def __init__(self):
        super(wrt_neuralNetwork_1, self).__init__()

        #
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # An affine operation: y = wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # Image dimensions 6 x 6
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        #
        x = F.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features += s
        return num_features










