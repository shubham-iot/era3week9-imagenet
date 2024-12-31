import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1  # Expansion factor for the block

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Batch normalization
        self.relu = nn.ReLU(inplace=True)  # ReLU activation
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # Batch normalization
        self.downsample = downsample  # Downsample layer for skip connection

    def forward(self, x):
        identity = x  # Save the input for the skip connection
        out = self.conv1(x)  # Forward pass through the first layer
        out = self.bn1(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation
        out = self.conv2(out)  # Forward pass through the second layer
        out = self.bn2(out)  # Batch normalization

        if self.downsample is not None:
            identity = self.downsample(x)  # Apply downsampling if needed

        out += identity  # Add the skip connection
        out = self.relu(out)  # ReLU activation
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64  # Initial number of channels
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization
        self.relu = nn.ReLU(inplace=True)  # ReLU activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Max pooling layer
        # Create the four layers of the ResNet
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive average pooling
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # Fully connected layer for classification

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # Create a downsample layer if needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))  # Add the first block
        self.in_channels = out_channels * block.expansion  # Update the number of input channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))  # Add remaining blocks

        return nn.Sequential(*layers)  # Return the layer as a sequential model

    def forward(self, x):
        x = self.conv1(x)  # Forward pass through the initial convolution
        x = self.bn1(x)  # Batch normalization
        x = self.relu(x)  # ReLU activation
        x = self.maxpool(x)  # Max pooling
        x = self.layer1(x)  # Forward pass through layer 1
        x = self.layer2(x)  # Forward pass through layer 2
        x = self.layer3(x)  # Forward pass through layer 3
        x = self.layer4(x)  # Forward pass through layer 4
        x = self.avgpool(x)  # Adaptive average pooling
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.fc(x)  # Forward pass through the fully connected layer
        return x

def resnet50():
    return ResNet(BasicBlock, [3, 4, 6, 3])  # Create a ResNet-50 model
