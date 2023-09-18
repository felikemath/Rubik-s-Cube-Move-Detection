import torch.nn as nn
import torch

class SpatialCNN(nn.Module):
    def __init__(self):
        super(SpatialCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.full6 = nn.Linear(512 * 6 * 6, 4096)
        self.dropout1 = nn.Dropout()

        self.full7 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout()

        self.full8 = nn.Linear(2048, 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.pool2(out)

        out = self.conv3(out)

        out = self.conv4(out)

        out = self.conv5(out)
        out = self.pool3(out)

        out = out.view(out.size(0), -1)  # Flatten

        out = self.full6(out)
        out = self.dropout1(out)

        out = self.full7(out)
        out = self.dropout2(out)

        out = self.full8(out)
        # out = self.softmax(out)

        return out

class SpatialCNN2(nn.Module):
    def __init__(self):
        super(SpatialCNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.full6 = nn.Linear(256 * 31 * 31, 4096)
        self.dropout1 = nn.Dropout()

        self.full7 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout()

        self.full8 = nn.Linear(2048, 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.pool2(out)

        print(out.shape)
        out = out.view(out.size(0), -1)  # Flatten


        out = self.full6(out)
        out = self.dropout1(out)

        out = self.full7(out)
        out = self.dropout2(out)

        out = self.full8(out)
        # out = self.softmax(out)

        return out

class TemporalCNN(nn.Module):
    def __init__(self):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.full6 = nn.Linear(512 * 6 * 6, 4096)
        self.dropout1 = nn.Dropout()

        self.full7 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout()

        self.full8 = nn.Linear(2048, 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.pool2(out)

        out = self.conv3(out)

        out = self.conv4(out)

        out = self.conv5(out)
        out = self.pool3(out)

        out = out.view(out.size(0), -1)  # Flatten

        out = self.full6(out)
        out = self.dropout1(out)

        out = self.full7(out)
        out = self.dropout2(out)

        out = self.full8(out)
        out = self.softmax(out)

        return out

if __name__ == '__main__':

    model = SpatialCNN2()
    temporal_model = TemporalCNN()
    x = torch.rand((1, 3, 512, 512))
    x2 = torch.rand((1,10,224,224))

    output = model(x)
    print(output)
    print(output.shape)