import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstancyNetwork(nn.Module):
    """
    Class for the overall network used in the constancy experiments
    """
    def __init__(self, dropout_rate=0.1):
        """
        Initialization defines the layers within the network
        
        Args:
            dropout_rate: Probability of dropping units during training
        """
        super(ConstancyNetwork, self).__init__()

        # Set the network parameters
        self.histograms_per_image = 4
        self.filter_size = 64
        self.kernel_size = 5
        self.stride = 2
        self.pool_kernel = 2
        self.pool_stride = 2
        self.convolutions_post_concat = 2
        self.convolutions_post_pool = 3
        self.bucket_size = 128
        self.dropout_rate = dropout_rate

        # Determine padding size
        if self.kernel_size == 3:
            self.padding = 1
        if self.kernel_size == 5:
            self.padding = (2, 2)

        # Determine input features to linear layer (divided by 4 because always doing a convolution and pool)
        self.linear_dimensions = int(self.bucket_size / 4 / (2 ** self.convolutions_post_pool) /
                                     (2 ** self.convolutions_post_concat))

        # Layers of the network for each individual histogram
        self.double_conv_layers = nn.ModuleList()
        for i in range(self.histograms_per_image):
            self.double_conv_layers.append(_DoubleConvPool1D(1, self.padding, self.dropout_rate))

        # Final layers for concatenated outputs and output
        self.conv2d_1 = nn.Conv2d(in_channels=self.filter_size * self.histograms_per_image,
                                  out_channels=self.filter_size, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding)
        self.bn1 = nn.BatchNorm2d(self.filter_size)
        self.dropout1 = nn.Dropout2d(self.dropout_rate)
        
        self.conv2d_2 = nn.ModuleList()
        for i in range(self.convolutions_post_concat - 1):
                self.conv2d_2.append(nn.Conv2d(in_channels=self.filter_size, out_channels=self.filter_size,
                                  kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
        
        self.bn2 = nn.BatchNorm2d(self.filter_size)
        self.dropout2 = nn.Dropout2d(self.dropout_rate)
        
        self.flatten = nn.Flatten()
        self.fc_output = nn.Linear(in_features=self.filter_size * (self.linear_dimensions ** 2), out_features=3)
        self.dropout_fc = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        """
        Defines the forward pass for the entire constancy network
        :param x: The input (four histograms)
        :return: The three-channel mean illuminant-color estimation distributions
        """
        # Get the image histograms
        histograms = [x[:, i:i+1] for i in range(self.histograms_per_image)]

        # Run the double convolution block for each histogram
        # No activations as this happens in the block
        for i in range(len(histograms)):
            histograms[i] = self.double_conv_layers[i](histograms[i])

        # Concatenate the projected convolutions
        x_concat = torch.cat(histograms, dim=1)

        # Apply first convolution on concatenated convolution with batch norm and dropout
        x_concat = self.conv2d_1(x_concat)
        x_concat = self.bn1(x_concat)
        x_concat = F.relu(x_concat)
        x_concat = self.dropout1(x_concat)

        # Apply the desired number of convolutions, if additional convolutions were specified.
        # Subtract one to make consistent with total convolutions count post concatenation
        if self.convolutions_post_concat > 1:
            for i in range(self.convolutions_post_concat - 1):
                x_concat = self.conv2d_2[i](x_concat)
                # Add safety check for BatchNorm (when features become 1x1)
                if x_concat.size(2) > 1 and x_concat.size(3) > 1:
                    x_concat = self.bn2(x_concat)
                x_concat = F.relu(x_concat)
                x_concat = self.dropout2(x_concat)

        # Flatten and get output
        x_concat = self.flatten(x_concat)
        x_concat = self.dropout_fc(x_concat)
        x_concat = F.sigmoid(self.fc_output(x_concat))
        return x_concat


class _DoubleConvPool1D(nn.Module):
    """
    Defines a standard subset of the constancy network that each histogram will separately use
    """
    def __init__(self, in_channels, padding, dropout_rate=0.3):
        """
        Initializes the network layer subset
        :param in_channels:  The number of channels being input to the first convolution
        :param padding:      The padding to use for convolutions
        :param dropout_rate: Probability of dropping units during training
        """
        super(_DoubleConvPool1D, self).__init__()

        # Set parameters
        self.histograms_per_image = 4
        self.filter_size = 64
        self.kernel_size = 5
        self.stride = 2
        self.pool_kernel = 2
        self.pool_stride = 2
        self.convolutions_post_concat = 2
        self.convolutions_post_pool = 3
        self.dropout_rate = dropout_rate

        # Layers of network block
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=self.filter_size,
                                  kernel_size=self.kernel_size, stride=self.stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(self.filter_size)
        self.dropout1 = nn.Dropout2d(self.dropout_rate)
        
        self.pool = nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        self.conv2d_2 = nn.ModuleList()
        for i in range(self.convolutions_post_pool):
            self.conv2d_2.append(nn.Conv2d(in_channels=self.filter_size, out_channels=self.filter_size,
                                  kernel_size=self.kernel_size, stride=self.stride, padding=padding))
        self.bn2 = nn.BatchNorm2d(self.filter_size)
        self.dropout2 = nn.Dropout2d(self.dropout_rate)

    def forward(self, x):
        """
        Defines the forward pass for the network block
        :param x: The input (will be a single histogram)
        :return: The result of the double convolution with intermediary pooling
        """
        x = self.conv2d_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.pool(x)
        
        for i in range(self.convolutions_post_pool):
            x = self.conv2d_2[i](x)
            # Add safety check for BatchNorm (when features become 1x1)
            if x.size(2) > 1 and x.size(3) > 1:
                x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout2(x)
            
        return x