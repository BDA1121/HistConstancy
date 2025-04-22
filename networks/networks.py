import torch
import torch.nn as nn
from torchvision.models import googlenet
import math
import torch.nn.functional as F
def radians_to_degrees(radians):
    """
    Convert radians to degrees
    :param radians: Tensor containing values in radians
    :return: Tensor with values converted to degrees
    """
    return radians * 180.0 / math.pi

def get_angular_error(output, ground_truth):
    """
    Gets the angular error between an output and a target with normalization
    :param output: The output (predicted) vector
    :param ground_truth: The target (actual) vector
    :return: The angular error between the two vectors
    """
    # Set both output and target to double to avoid dtype overflow issues
    output = output.type(torch.double)
    ground_truth = ground_truth.type(torch.double)
    
    # # Normalize the vectors
    # output_normalized = F.normalize(output, p=2, dim=1)
    # ground_truth_normalized = F.normalize(ground_truth,p=2,dim=1)
    
    # Set up cosine similarity
    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # Get cosine similarity scores
    cos_sim_errors = cosine_similarity(output, ground_truth)
    # Take inverse cosine to get angular errors
    angular_errors_radians = torch.arccos(cos_sim_errors)
    angular_errors_degrees = radians_to_degrees(angular_errors_radians)
    return angular_errors_degrees

def get_euclidean_distance(output, ground_truth):
    """
    Calculates the Euclidean distance between two tensors
    :param output:       The predicted illuminations
    :param ground_truth: The ground truth illuminations
    :return: The average for the batch euclidean distance
    """
    differences = torch.sub(output, ground_truth)
    squared = torch.pow(differences, exponent=2)
    summed = torch.sum(squared, dim=1)
    rooted = torch.sqrt(summed)
    return torch.mean(rooted)

def combined_loss(output, target, angular_weight=0.02    , euclidean_weight=0.5):
    """
    Combined loss function with weighted angular and euclidean components
    
    :param output: The predicted illuminations
    :param target: The ground truth illuminations
    :param angular_weight: Weight for the angular component (default: 0.6)
    :param euclidean_weight: Weight for the euclidean component (default: 0.4)
    :return: Combined weighted loss
    """
    # Calculate angular error
    angular_error = torch.mean(get_angular_error(output, target))
    
    # Calculate euclidean distance
    euclidean_error = get_euclidean_distance(output, target)
    
    # Combine the losses with weights
    combined = angular_weight * angular_error + euclidean_weight * euclidean_error
    # print(f'angular: {angular_error}---------euclidean: {euclidean_error}---------------combined:{combined}')
    
    return combined

# Define GoogleNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import googlenet

class CustomGoogleNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomGoogleNet, self).__init__()
        # Create a modified GoogleNet that takes 1 channel input
        self.googlenet = googlenet(pretrained=True)
        
        # Modify the first convolutional layer to accept 1 channel instead of 3
        # Save the original weights for initialization
        original_conv = self.googlenet.conv1.conv
        original_weights = original_conv.weight.data
        
        # Create a new conv layer with 1 input channel
        self.googlenet.conv1.conv = nn.Conv2d(
            1, 
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize the new conv layer with the average of the original weights across the channel dimension
        with torch.no_grad():
            # Average the weights across the input channels (dim=1)
            new_weights = original_weights.mean(dim=1, keepdim=True)
            self.googlenet.conv1.conv.weight.copy_(new_weights)
            if original_conv.bias is not None:
                self.googlenet.conv1.conv.bias.copy_(original_conv.bias)
        
        # Get the feature dimension
        num_ftrs = self.googlenet.fc.in_features
        
        # Remove the final fully connected layer
        self.googlenet.fc = nn.Identity()
        
        # Create a flatten layer
        self.flatten = nn.Flatten()
        
        # Final FC layer for the concatenated features
        # For multi-channel input with each channel processed separately
        self.fc_output = nn.Linear(num_ftrs * 4, num_classes)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # List to store outputs from each channel
        channel_outputs = []
        
        # Process each channel separately
        for i in range(channels):
            # Extract single channel without repeating
            # Shape becomes [batch_size, 1, height, width]
            x_channel = x[:, i:i+1, :, :].repeat(1, 3, 1, 1)
            
            # Pass through modified GoogleNet backbone
            output = self.googlenet(x_channel)
            channel_outputs.append(output)
        
        # Concatenate all channel outputs
        x_concat = torch.cat(channel_outputs, dim=1)
        
        # Flatten and pass through final FC layer with sigmoid activation
        x_concat = self.flatten(x_concat)
        x_concat = F.sigmoid(self.fc_output(x_concat))
        
        return x_concat

def get_loss_function(loss_type):
    """
    Return the appropriate loss function based on the loss type
    :param loss_type: String indicating which loss function to use ('mse', 'angular', 'euclidean')
    :return: Loss function
    """
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'angular':
        return lambda output, target: torch.mean(get_angular_error(output, target))
    elif loss_type == 'euclidean':
        return get_euclidean_distance
    elif loss_type == 'combined':
        return combined_loss
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")