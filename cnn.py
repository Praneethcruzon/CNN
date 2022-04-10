import torch.nn as nn
from collections import OrderedDict
# nn class defines all modules required for Neural Network training
# All network that we define should be a derived class of nn class. 

# Initializing CNN as a derived class of NN.
class CNN(nn.Module):
    def __init__(self, input_channels):

        self.features = 32
        # Referring to __init__ of base nn class.
        super().__init__()

        # Create blocks from _block() template. 
        self.conv1 = self._block(input_channels, self.features, "Block 1")
        self.conv2 = self._block(self.features, self.features * 2, "Block 2")
        self.conv3 = self._block(self.features * 2, self.features * 4, "Block 3")
        self.conv4 = self._block(self.features * 4, self.features * 8, "Block 4")
        self.conv5 = self._block(self.features * 8, self.features * 16, "Block 5")

        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 1)

    def _block(self, in_channels, features, name):
        # Creates a template of block containing Con2D, BatchNorm and Activation functions

        # Cov2d Parameters : in_channels, out_channels, kernel_size, stride=1, 
        # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
        # device=None, dtype=None
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + " - Convolution 2D",
                        nn.Conv2d(
                            in_channels = in_channels, 
                            out_channels = features, 
                            kernel_size = 3,
                            padding = 1
                        )
                    ),
                    (
                        name + " - Batch Normalization",
                        nn.BatchNorm2d(num_features = features)
                    ),
                    (
                        name + " - ReLU Activation",
                        nn.ReLU()
                    )
                ]
            )
        )
        

    # Forward method of Pytorch NN module defines the computation 
    # performed at every call. This function should be overriden.   
    def forward(self, x):

        print(f"before conv {x.shape}")
        x = self.conv1(x)
        x = self.max_pool(x)
        print(f"After 1 conv {x.shape}")

        x = self.conv2(x)
        x = self.max_pool(x)
        print(f"After 2 conv {x.shape}")

        x = self.conv3(x)
        x = self.max_pool(x)
        print(f"After 3 conv {x.shape}")

        x = self.conv4(x)
        x = self.max_pool(x)
        print(f"After 4 conv {x.shape}")

        x = self.conv5(x)
        x = self.max_pool(x)
        print(f"After 5 conv {x.shape}")

        return x

