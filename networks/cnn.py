import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstancyNetwork(nn.Module):
    """
    Enhanced class for the overall network used in the constancy experiments
    with dynamic dimension calculation
    """
    def __init__(self, dropout_rate=0.2):
        """
        Initialization defines the layers within the network
        """
        super(ConstancyNetwork, self).__init__()

        # Set the network parameters
        self.histograms_per_image = 4
        self.filter_size = 96 
        self.kernel_size = 5
        self.stride = 1  
        self.pool_kernel = 2
        self.pool_stride = 2
        self.convolutions_post_concat = 2  
        self.convolutions_post_pool = 2  
        self.bucket_size = 128
        self.dropout_rate = dropout_rate

        # Determine padding size
        self.padding = (2, 2)  # Always use same padding

        # Layers of the network for each individual histogram
        self.double_conv_layers = nn.ModuleList()
        for i in range(self.histograms_per_image):
            self.double_conv_layers.append(_DoubleConvPool1D(1, self.padding, self.filter_size, 
                                                            self.kernel_size, self.stride, 
                                                            self.pool_kernel, self.pool_stride,
                                                            self.convolutions_post_pool))

        # Final layers for concatenated outputs and output
        self.conv2d_1 = nn.Conv2d(in_channels=self.filter_size * self.histograms_per_image,
                                  out_channels=self.filter_size, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding)
        self.batch_norm1 = nn.BatchNorm2d(self.filter_size)
        
        self.conv2d_2 = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(self.convolutions_post_concat - 1):
            self.conv2d_2.append(nn.Conv2d(in_channels=self.filter_size, out_channels=self.filter_size,
                                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            self.batch_norms.append(nn.BatchNorm2d(self.filter_size))
        
        # Add global average pooling instead of flattening
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)
        self.flatten = nn.Flatten()
        
        # Adaptive fully connected layers
        fc_input_size = self.filter_size * 4 * 4  # From global avg pooling
        fc_intermediate_size = 256
        self.fc_intermediate = nn.Linear(in_features=fc_input_size, out_features=fc_intermediate_size)
        self.fc_output = nn.Linear(in_features=fc_intermediate_size, out_features=3)

    def forward(self, x):
        """
        Defines the forward pass for the entire constancy network
        :param x: The input (four histograms)
        :return: The three-channel mean illuminant-color estimation distributions
        """
        # Get the image histograms
        histograms = [x[:, i:i+1] for i in range(self.histograms_per_image)]

        # Run the double convolution block for each histogram
        for i in range(len(histograms)):
            histograms[i] = self.double_conv_layers[i](histograms[i])

        # Concatenate the projected convolutions
        x_concat = torch.concatenate(histograms, dim=1)

        # Apply first convolution on concatenated convolution with batch normalization
        x_concat = self.conv2d_1(x_concat)
        # x_concat = self.batch_norm1(x_concat)
        x_concat = F.relu(x_concat)

        # Apply the desired number of convolutions, if additional convolutions were specified.
        if self.convolutions_post_concat > 1:
            for i in range(self.convolutions_post_concat - 1):
                x_concat = self.conv2d_2[i](x_concat)
                # x_concat = self.batch_norms[i](x_concat)
                x_concat = F.relu(x_concat)

        # Apply global average pooling to get fixed dimensions
        x_concat = self.global_avg_pool(x_concat)
        
        # Flatten and get output
        x_concat = self.flatten(x_concat)
        
        # Add intermediate fully connected layer with dropout
        x_concat = F.relu(self.fc_intermediate(x_concat))
        # x_concat = self.dropout(x_concat)
        
        # Final output layer
        x_concat = F.sigmoid(self.fc_output(x_concat))
        return x_concat


class _DoubleConvPool1D(nn.Module):
    """
    Defines a standard subset of the constancy network that each histogram will separately use
    """
    def __init__(self, in_channels, padding, filter_size, kernel_size, stride, pool_kernel, pool_stride, convolutions_post_pool):
        """
        Initializes the network layer subset
        :param in_channels: The number of channels being input to the first convolution
        :param padding: Padding value for the convolutions
        """
        super(_DoubleConvPool1D, self).__init__()

        # Set parameters from passed values
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.convolutions_post_pool = convolutions_post_pool

        # Layers of network block
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=self.filter_size // 2,
                                  kernel_size=self.kernel_size, stride=self.stride, padding=padding)
        self.batch_norm1 = nn.BatchNorm2d(self.filter_size // 2)
        self.pool = nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        
        # First layer post pool with different input/output channels
        self.conv_post_pool_first = nn.Conv2d(in_channels=self.filter_size // 2, 
                                             out_channels=self.filter_size,
                                             kernel_size=self.kernel_size, 
                                             stride=self.stride, 
                                             padding=padding)
        self.batch_norm_first = nn.BatchNorm2d(self.filter_size)
        
        self.conv2d_2 = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Remaining layers post pool (if any)
        for i in range(self.convolutions_post_pool - 1):
            self.conv2d_2.append(nn.Conv2d(in_channels=self.filter_size, 
                                          out_channels=self.filter_size,
                                          kernel_size=self.kernel_size, 
                                          stride=self.stride, 
                                          padding=padding))
            self.batch_norms.append(nn.BatchNorm2d(self.filter_size))

    def forward(self, x):
        """
        Defines the forward pass for the network block
        :param x: The input (will be a single histogram)
        :return: The result of the double convolution with intermediary pooling
        """
        x = self.conv2d_1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # First conv after pooling
        x = self.conv_post_pool_first(x)
        # x = self.batch_norm_first(x)
        x = F.relu(x)
        
        # Remaining convs (if any)
        for i in range(self.convolutions_post_pool - 1):
            x = self.conv2d_2[i](x)
            # x = self.batch_norms[i](x)
            x = F.relu(x)
            
        return x
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ConstancyNetwork(nn.Module):
#     """
#     Enhanced class for the overall network used in the constancy experiments
#     with dynamic dimension calculation
#     """
#     def __init__(self, dropout_rate=0.2):
#         """
#         Initialization defines the layers within the network
#         """
#         super(ConstancyNetwork, self).__init__()

#         # Set the network parameters
#         self.histograms_per_image = 4
#         self.filter_size = 96  # Adjusted filter size
#         self.kernel_size = 5
#         self.stride = 1  # Reduced stride to maintain spatial dimensions
#         self.pool_kernel = 2
#         self.pool_stride = 2
#         self.convolutions_post_concat = 2  # Reduced to avoid over-shrinking
#         self.convolutions_post_pool = 2  # Reduced to avoid over-shrinking
#         self.bucket_size = 128
#         self.dropout_rate = dropout_rate

#         # Determine padding size
#         self.padding = (2, 2)  # Always use same padding

#         # Layers of the network for each individual histogram
#         self.double_conv_layers = nn.ModuleList()
#         for i in range(self.histograms_per_image):
#             self.double_conv_layers.append(_DoubleConvPool1D(1, self.padding, self.filter_size, 
#                                                             self.kernel_size, self.stride, 
#                                                             self.pool_kernel, self.pool_stride,
#                                                             self.convolutions_post_pool))

#         # Final layers for concatenated outputs and output
#         self.conv2d_1 = nn.Conv2d(in_channels=self.filter_size * self.histograms_per_image,
#                                   out_channels=self.filter_size, kernel_size=self.kernel_size,
#                                   stride=self.stride, padding=self.padding)
#         self.batch_norm1 = nn.BatchNorm2d(self.filter_size)
        
#         self.conv2d_2 = nn.ModuleList()
#         self.batch_norms = nn.ModuleList()
        
#         for i in range(self.convolutions_post_concat - 1):
#             self.conv2d_2.append(nn.Conv2d(in_channels=self.filter_size, out_channels=self.filter_size,
#                                 kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
#             self.batch_norms.append(nn.BatchNorm2d(self.filter_size))
        
#         # Add global average pooling instead of flattening
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        
#         # Add dropout for regularization
#         self.dropout = nn.Dropout(self.dropout_rate)
#         self.flatten = nn.Flatten()
        
#         # Adaptive fully connected layers
#         fc_input_size = self.filter_size * 4 * 4  # From global avg pooling
#         fc_intermediate_size = 256
#         self.fc_intermediate = nn.Linear(in_features=fc_input_size, out_features=fc_intermediate_size)
#         self.fc_output = nn.Linear(in_features=fc_intermediate_size, out_features=3)

#     def forward(self, x):
#         """
#         Defines the forward pass for the entire constancy network
#         :param x: The input (four histograms)
#         :return: The three-channel mean illuminant-color estimation distributions
#         """
#         # Get the image histograms
#         histograms = [x[:, i:i+1] for i in range(self.histograms_per_image)]

#         # Run the double convolution block for each histogram
#         for i in range(len(histograms)):
#             histograms[i] = self.double_conv_layers[i](histograms[i])

#         # Concatenate the projected convolutions
#         x_concat = torch.concatenate(histograms, dim=1)

#         # Apply first convolution on concatenated convolution with batch normalization
#         x_concat = self.conv2d_1(x_concat)
#         x_concat = self.batch_norm1(x_concat)
#         x_concat = F.relu(x_concat)

#         # Apply the desired number of convolutions, if additional convolutions were specified.
#         if self.convolutions_post_concat > 1:
#             for i in range(self.convolutions_post_concat - 1):
#                 x_concat = self.conv2d_2[i](x_concat)
#                 x_concat = self.batch_norms[i](x_concat)
#                 x_concat = F.relu(x_concat)

#         # Apply global average pooling to get fixed dimensions
#         x_concat = self.global_avg_pool(x_concat)
        
#         # Flatten and get output
#         x_concat = self.flatten(x_concat)
        
#         # Add intermediate fully connected layer with dropout
#         x_concat = F.relu(self.fc_intermediate(x_concat))
#         x_concat = self.dropout(x_concat)
        
#         # Final output layer
#         x_concat = F.sigmoid(self.fc_output(x_concat))
#         return x_concat


# class _DoubleConvPool1D(nn.Module):
#     """
#     Defines a standard subset of the constancy network that each histogram will separately use
#     """
#     def __init__(self, in_channels, padding, filter_size, kernel_size, stride, pool_kernel, pool_stride, convolutions_post_pool):
#         """
#         Initializes the network layer subset
#         :param in_channels: The number of channels being input to the first convolution
#         :param padding: Padding value for the convolutions
#         """
#         super(_DoubleConvPool1D, self).__init__()

#         # Set parameters from passed values
#         self.filter_size = filter_size
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.pool_kernel = pool_kernel
#         self.pool_stride = pool_stride
#         self.convolutions_post_pool = convolutions_post_pool

#         # Layers of network block
#         self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=self.filter_size // 2,
#                                   kernel_size=self.kernel_size, stride=self.stride, padding=padding)
#         self.batch_norm1 = nn.BatchNorm2d(self.filter_size // 2)
#         self.pool = nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        
#         # First layer post pool with different input/output channels
#         self.conv_post_pool_first = nn.Conv2d(in_channels=self.filter_size // 2, 
#                                              out_channels=self.filter_size,
#                                              kernel_size=self.kernel_size, 
#                                              stride=self.stride, 
#                                              padding=padding)
#         self.batch_norm_first = nn.BatchNorm2d(self.filter_size)
        
#         self.conv2d_2 = nn.ModuleList()
#         self.batch_norms = nn.ModuleList()
        
#         # Remaining layers post pool (if any)
#         for i in range(self.convolutions_post_pool - 1):
#             self.conv2d_2.append(nn.Conv2d(in_channels=self.filter_size, 
#                                           out_channels=self.filter_size,
#                                           kernel_size=self.kernel_size, 
#                                           stride=self.stride, 
#                                           padding=padding))
#             self.batch_norms.append(nn.BatchNorm2d(self.filter_size))

#     def forward(self, x):
#         """
#         Defines the forward pass for the network block
#         :param x: The input (will be a single histogram)
#         :return: The result of the double convolution with intermediary pooling
#         """
#         x = self.conv2d_1(x)
#         x = self.batch_norm1(x)
#         x = F.relu(x)
#         x = self.pool(x)
        
#         # First conv after pooling
#         x = self.conv_post_pool_first(x)
#         x = self.batch_norm_first(x)
#         x = F.relu(x)
        
#         # Remaining convs (if any)
#         for i in range(self.convolutions_post_pool - 1):
#             x = self.conv2d_2[i](x)
#             x = self.batch_norms[i](x)
#             x = F.relu(x)
            
#         return x