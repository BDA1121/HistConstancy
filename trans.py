import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from dataloader import HistogramDataset
from multidataloader import HistogramDataset
from cnn import ConstancyNetwork
from networks import CustomGoogleNet, get_loss_function
from tqdm import tqdm
import argparse
import os
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Enable cuDNN benchmarking for performance
torch.backends.cudnn.benchmark = True

# Clear CUDA cache before starting
torch.cuda.empty_cache()


# Add the HistogramTransformer model
class HistogramTransformer(nn.Module):
    def __init__(self, input_height, input_width, num_classes=3, num_layers=2, 
                 dim_feedforward=32, nhead=3, dropout=0.1):
        super(HistogramTransformer, self).__init__()
        
        # Model dimensions
        self.input_height = input_height
        self.input_width = input_width
        self.d_model = 32  # Embedding dimension
        
        # Channel names for readability/reference
        self.channel_names = ['rg_hist', 'gb_hist', 'br_hist', 'plane_proj_hist']
        
        # Position encoding
        self.position_encoding = self.create_position_encoding(input_height, input_width, self.d_model)
        
        # Initial projection layer to convert histogram values to embedding dimension
        self.embedding = nn.Linear(1, self.d_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers
        self.flatten = nn.Flatten()
        # For 4 channels
        self.fc_output = nn.Linear(self.d_model * input_height * input_width * 4, num_classes)
        
    def create_position_encoding(self, height, width, d_model):
        # Create 2D positional encoding
        position_encoding = torch.zeros(height * width, d_model)
        position = torch.arange(0, height * width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(position_encoding.unsqueeze(0), requires_grad=False)
    
    # Regular method instead of jit.script_method
    def process_channel(self, x_channel):
        # Project to embedding dimension
        x_embedded = self.embedding(x_channel)
        
        # Add positional encoding
        x_embedded = x_embedded + self.position_encoding
        
        # Pass through transformer encoder
        x_transformed = self.transformer_encoder(x_embedded)
        
        return x_transformed
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Check if we have the expected number of channels
        assert channels == 4, f"Expected 4 channels (RG, GB, BR, Plane Proj), but got {channels}"
        
        # List to store outputs from each channel
        channel_outputs = []
        
        # Process each channel separately
        for i in range(channels):
            # Extract single channel and reshape to [batch_size, height*width, 1]
            x_channel = x[:, i:i+1, :, :].reshape(batch_size, height * width, 1)
            
            # Process channel using optimized method
            x_transformed = self.process_channel(x_channel)
            
            # Flatten to [batch_size, d_model * height * width]
            x_flat = x_transformed.reshape(batch_size, -1)
            
            # Store the output
            channel_outputs.append(x_flat)
        
        # Concatenate all channel outputs
        x_concat = torch.cat(channel_outputs, dim=1)
        
        # Final FC layer with sigmoid activation
        x_concat = torch.sigmoid(self.fc_output(x_concat))
        
        return x_concat


# Fast Attention mechanism for improved transformer performance
class FastAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        b, n, d = x.shape
        h = self.heads
        
        # Get query, key, value
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        
        # Compute attention efficiently
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


def train_model(model, dataloader, criterion, optimizer, num_epochs=10, save_dir='checkpoints', 
                save_interval=5, model_name='model', fp16=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if fp16 else None
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Track best model
    best_loss = float('inf')
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for inputs, targets in progress_bar:
            inputs = inputs.to(device, non_blocking=True)  # non_blocking for async transfer
            targets = targets.to(device, non_blocking=True)

            # More efficient gradient zeroing
            optimizer.zero_grad(set_to_none=True)

            if fp16:
                # Mixed precision training path
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training path
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # Update progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Update learning rate
        scheduler.step(epoch_loss)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'scaler': scaler.state_dict() if fp16 else None,
            }, best_model_path)
            print(f"New best model saved with loss: {epoch_loss:.4f}")
        
        # Save model checkpoint at specified intervals
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'scaler': scaler.state_dict() if fp16 else None,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training complete!")
    print(f"Best model saved with loss: {best_loss:.4f}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train model with different loss functions')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'angular', 'euclidean','combined'],
                        help='Loss function to use: mse, angular, or euclidean')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save model checkpoints every N epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--transform', type=str, default='None', choices=['tan', 'log', 'None'],
                        help='Transform function for histogram')
    parser.add_argument('--net', type=str, default='None', choices=['google', 'color', 'transformer'],
                        help='Network architecture to use')
    parser.add_argument('--hist_size', type=int, default=64,
                        help='Size of histogram (height/width)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training (faster but may impact accuracy)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches to prefetch per worker')
    
    args = parser.parse_args()
    
    # Print training configuration
    print(f"Using {args.loss} loss function")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Saving checkpoints every {args.save_interval} epochs to {args.save_dir}")
    print(f"Network architecture: {args.net}")
    print(f"Mixed precision training: {'Enabled' if args.fp16 else 'Disabled'}")
    print(f"Number of data loading workers: {args.num_workers}")
    
    # Paths setup
    image_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_training"
    depth_dir = "/work/SuperResolutionData/spectralRatio/data/depth_for_training"
    normal_dir = "/work/SuperResolutionData/spectralRatio/data/surface_norm_for_training"
    csv_file = "/home/balamurugan.d/src/train.csv"
    
    folders = [f"folder_{i}" for i in range(1, 10)] 

    # Create dataset
    dataset = HistogramDataset(image_dir, depth_dir, normal_dir, csv_file, folders=folders, plane=True, transform=args.transform)
    
    # Optimized DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )

    # Initialize the model based on the specified architecture
    if args.net == 'google':
        model = CustomGoogleNet(num_classes=3)
        model_name = 'googlenet'
    elif args.net == 'transformer':
        model = HistogramTransformer(
            input_height=args.hist_size,
            input_width=args.hist_size,
            num_classes=3,
            num_layers=3,
            dim_feedforward=512,
            nhead=8,
            dropout=0.1
        )
        model_name = 'transformer'
    else:  # Default to ConstancyNetwork
        model = ConstancyNetwork()
        model_name = 'constancy'
    
    # Initialize criterion and optimizer
    criterion = get_loss_function(args.loss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model with optimizations
    train_model(
        model, 
        dataloader, 
        criterion, 
        optimizer, 
        num_epochs=args.epochs, 
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        model_name=model_name,
        fp16=args.fp16
    )

    # Save the final trained model using efficient serialization
    final_model_path = os.path.join(args.save_dir, f"{model_name}_trained_{args.loss}.pth")
    torch.save(model.state_dict(), final_model_path, _use_new_zipfile_serialization=True)
    print(f"Final model saved to {final_model_path}")