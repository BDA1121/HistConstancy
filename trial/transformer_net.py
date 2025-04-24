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

# Add the HistogramTransformer model
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Module to convert histograms into patch embeddings for transformer input
    """
    def __init__(self, hist_size=128, patch_size=16, embed_dim=256, in_channels=1):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (hist_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for transformer blocks
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """
    MLP block for transformer
    """
    def __init__(self, embed_dim, mlp_ratio=4.0):
        super(MLP, self).__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard transformer block with pre-norm architecture
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HistogramViT(nn.Module):
    """
    Vision Transformer for histogram-based color constancy
    """
    def __init__(self, hist_size=128, patch_size=16, embed_dim=256, 
                 depth=6, num_heads=8, mlp_ratio=4.0, histograms_per_image=4):
        super(HistogramViT, self).__init__()
        
        self.histograms_per_image = histograms_per_image
        
        # Histogram embedding layers for each histogram
        self.patch_embed_layers = nn.ModuleList([
            PatchEmbedding(hist_size, patch_size, embed_dim) 
            for _ in range(histograms_per_image)
        ])
        
        # Class token for each histogram stream
        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, embed_dim))
            for _ in range(histograms_per_image)
        ])
        
        # Position embeddings for each histogram stream
        num_patches = (hist_size // patch_size) ** 2
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for cls token
            for _ in range(histograms_per_image)
        ])
        
        # Transformer blocks for each histogram stream
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio) 
                for _ in range(depth)
            ])
            for _ in range(histograms_per_image)
        ])
        
        # Fusion transformer blocks to combine histogram features
        self.fusion_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(3)  # 3 fusion transformer blocks
        ])
        
        # Final MLP head for RGB prediction
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * histograms_per_image, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Sigmoid()  # RGB values should be between 0 and 1
        )
        
        # Initialize position embeddings
        for pos_embed in self.pos_embeds:
            nn.init.trunc_normal_(pos_embed, std=0.02)
            
        # Initialize class tokens
        for cls_token in self.cls_tokens:
            nn.init.trunc_normal_(cls_token, std=0.02)
            
    def forward(self, x):
        # Split input into individual histograms
        histograms = [x[:, i:i+1] for i in range(self.histograms_per_image)]
        
        # Process each histogram through its own transformer stream
        cls_tokens_out = []
        for i in range(self.histograms_per_image):
            # Embed patches
            x_embed = self.patch_embed_layers[i](histograms[i])
            
            # Add class token
            cls_token = self.cls_tokens[i].expand(x_embed.shape[0], -1, -1)
            x_embed = torch.cat((cls_token, x_embed), dim=1)
            
            # Add position embeddings
            x_embed = x_embed + self.pos_embeds[i]
            
            # Apply transformer blocks
            for block in self.blocks[i]:
                x_embed = block(x_embed)
                
            # Get CLS token output
            cls_tokens_out.append(x_embed[:, 0])
        
        # Concatenate CLS tokens from all streams
        combined_features = torch.cat(cls_tokens_out, dim=1)
        
        # Final prediction
        rgb_pred = self.head(combined_features)
        
        return rgb_pred


# # Example usage:
# if __name__ == "__main__":
#     # Create a batch of 4 histograms with size 128x128 for 2 images
#     batch_size = 2
#     histograms_per_image = 4
#     hist_size = 128
    
#     # Create random input as an example
#     dummy_input = torch.randn(batch_size, histograms_per_image, 1, hist_size, hist_size)
    
#     # Initialize model
#     model = HistogramViT(
#         hist_size=hist_size,
#         patch_size=16,
#         embed_dim=256,
#         depth=6,
#         num_heads=8,
#         histograms_per_image=histograms_per_image
#     )
    
#     # Forward pass
#     output = model(dummy_input)
    
#     print(f"Input shape: {dummy_input.shape}")
#     print(f"Output shape: {output.shape}")  # Should be [batch_size, 3]
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from multidataloader import HistogramDataset
from cnn import ConstancyNetwork
from networks import CustomGoogleNet, get_loss_function
from tqdm import tqdm
import argparse
import os
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts


def init_weights(m):
    """Initialize model weights properly based on layer type"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=10, 
                save_dir='checkpoints', save_interval=5, model_name='model', patience=10,
                weight_decay=1e-4, gradient_clip=1.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # For early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")
    
    # For plotting
    train_losses = []
    val_losses = []
    epochs_list = []
    learning_rates = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        batch_train_losses = []
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (inputs, targets) in enumerate(train_progress):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Add L2 regularization manually if using custom loss function
            if weight_decay > 0:
                l2_reg = 0.0
                for param in model.parameters():
                    l2_reg += torch.norm(param, p=2)
                loss += weight_decay * l2_reg
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
            optimizer.step()

            batch_loss = loss.item()
            running_train_loss += batch_loss * inputs.size(0)
            batch_train_losses.append(batch_loss)
            
            # Update progress bar with current loss and learning rate
            if scheduler is not None and isinstance(scheduler, CosineAnnealingWarmRestarts):
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
                
            train_progress.set_postfix(
                loss=batch_loss, 
                avg_loss=np.mean(batch_train_losses[-10:]) if batch_train_losses else 0,
                lr=current_lr
            )

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        batch_val_losses = []
        
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for inputs, targets in val_progress:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                batch_loss = loss.item()
                running_val_loss += batch_loss * inputs.size(0)
                batch_val_losses.append(batch_loss)
                
                val_progress.set_postfix(
                    loss=batch_loss,
                    avg_loss=np.mean(batch_val_losses[-10:]) if batch_val_losses else 0
                )
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        # Update learning rate scheduler if using ReduceLROnPlateau
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()
        
        # Store losses for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        epochs_list.append(epoch + 1)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Plot and save training curve after each epoch
        plot_losses(epochs_list, train_losses, val_losses, learning_rates, save_dir, model_name)
        
        # Save model checkpoint at specified intervals
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Check if this is the best model so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs_list': epochs_list,
                'learning_rates': learning_rates
            }, best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs (best val loss: {best_val_loss:.4f})")
            
        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs without improvement")
            break

    print("Training complete!")
    
    # Final plot with complete training history
    plot_losses(epochs_list, train_losses, val_losses, learning_rates, save_dir, model_name, final=True)
    
    return best_val_loss, train_losses, val_losses, epochs_list, learning_rates


def plot_losses(epochs, train_losses, val_losses, learning_rates, save_dir, model_name, final=False):
    """
    Plot training and validation losses with learning rate
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    ax1.set_title(f'{model_name} Training and Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Add min val loss annotation
    min_val_epoch = epochs[val_losses.index(min(val_losses))]
    min_val_loss = min(val_losses)
    ax1.annotate(f'Min Val Loss: {min_val_loss:.4f}',
                xy=(min_val_epoch, min_val_loss),
                xytext=(min_val_epoch + 1, min_val_loss + 0.02),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                fontsize=10)
    
    # Create a smooth y-axis scale
    y_min = min(min(train_losses), min(val_losses)) * 0.9
    y_max = min(max(train_losses[:5]), max(val_losses) * 1.5)  # Limit initial high train loss
    ax1.set_ylim([y_min, y_max])
    
    # Plot learning rate
    ax2.plot(epochs, learning_rates, 'g-', label='Learning Rate')
    ax2.set_yscale('log')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot with timestamp or final designation
    suffix = '_final' if final else ''
    plot_path = os.path.join(save_dir, f"{model_name}_loss_plot{suffix}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if final:
        print(f"Final loss plot saved to {plot_path}")


def plot_angular_error_histogram(errors, metrics, save_dir, model_name):
    """
    Plot histogram of angular errors with metric lines
    """
    plt.figure(figsize=(12, 8))
    
    # Plot histogram of errors
    n, bins, patches = plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add vertical lines for various metrics
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    metric_names = ['mean', 'median', 'trimean', 'best_25', 'worst_25']
    for i, metric in enumerate(metric_names):
        plt.axvline(x=metrics[metric], color=colors[i], linestyle='--', 
                   label=f"{metric.replace('_', ' ').title()}: {metrics[metric]:.2f}°")
    
    plt.title(f'Angular Error Distribution - {model_name}')
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the histogram
    hist_path = os.path.join(save_dir, f"{model_name}_error_histogram.png")
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Angular error histogram saved to {hist_path}")


def calculate_angular_error(pred, target):
    """
    Calculate the angular error between predicted and target RGB values.
    """
    device = pred.device
    
    # Normalize the vectors
    pred_norm = torch.nn.functional.normalize(pred, dim=1)
    target_norm = torch.nn.functional.normalize(target, dim=1)
    
    # Calculate the dot product
    dot_product = torch.sum(pred_norm * target_norm, dim=1)
    
    # Clamp to prevent numerical issues with arccos
    dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # Calculate angular error in degrees
    angular_error = torch.acos(dot_product) * (180.0 / math.pi)
    return angular_error


def evaluate_model(model, dataloader):
    """
    Evaluate the model on the given dataloader.
    Returns mean, median, trimean, best 25%, worst 25% angular errors.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    angular_errors = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            batch_errors = calculate_angular_error(outputs, targets)
            angular_errors.extend(batch_errors.cpu().numpy())
    
    angular_errors = np.array(angular_errors)
    
    # Calculate metrics
    mean_error = np.mean(angular_errors)
    median_error = np.median(angular_errors)
    trimean = 0.25 * (np.quantile(angular_errors, 0.25) + 2 * median_error + np.quantile(angular_errors, 0.75))
    best_25 = np.mean(np.sort(angular_errors)[:len(angular_errors)//4])
    worst_25 = np.mean(np.sort(angular_errors)[-(len(angular_errors)//4):])
    
    metrics = {
        'mean': mean_error,
        'median': median_error,
        'trimean': trimean,
        'best_25': best_25,
        'worst_25': worst_25
    }
    
    return metrics, angular_errors


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train model with different loss functions')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'angular', 'euclidean', 'combined'],
                        help='Loss function to use: mse, angular, euclidean, or combined')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save model checkpoints every N epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--transform', type=str, default='None', choices=['tan', 'log', 'None'],
                        help='Transform function for histogram')
    parser.add_argument('--net', type=str, default='transformer', choices=['google', 'color', 'transformer'],
                        help='Network architecture to use')
    parser.add_argument('--hist_size', type=int, default=128,
                        help='Size of histogram (height/width)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Percentage of data to use for validation (0-1)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size for ViT')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension for ViT')
    parser.add_argument('--depth', type=int, default=6,
                        help='Number of transformer layers for ViT')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads for ViT')
    parser.add_argument('--scheduler', type=str, default='reduce', choices=['reduce', 'cosine', 'none'],
                        help='Learning rate scheduler: reduce_on_plateau, cosine_annealing, or none')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty) for optimizer')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable)')
    parser.add_argument('--restart_from', type=str, default='',
                        help='Path to checkpoint to restart training from')
    
    args = parser.parse_args()
    
    image_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_training"
    depth_dir = "/work/SuperResolutionData/spectralRatio/data/depth_for_training"
    normal_dir = "/work/SuperResolutionData/spectralRatio/data/surface_norm_for_training"
    csv_file = "/home/balamurugan.d/src/train.csv"

    print(f"Using {args.loss} loss function")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    print(f"Initial learning rate: {args.lr}")
    print(f"Learning rate scheduler: {args.scheduler}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Gradient clipping: {args.gradient_clip}")
    print(f"Saving checkpoints every {args.save_interval} epochs to {args.save_dir}")
    print(f"Network architecture: {args.net}")
    print(f"Validation split: {args.val_split}")
    print(f"Early stopping patience: {args.patience}")
    
    if args.net == 'transformer':
        print(f"ViT parameters: patch_size={args.patch_size}, embed_dim={args.embed_dim}, depth={args.depth}, num_heads={args.num_heads}")
    
    folders = [f"folder_{i}" for i in range(1, 10)]

    # Create the dataset
    full_dataset = HistogramDataset(image_dir, depth_dir, normal_dir, csv_file, folders=folders, plane=True, transform=args.transform)
    
    # Split into training and validation sets
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

    # Initialize the model based on the specified architecture
    if args.net == 'google':
        model = CustomGoogleNet(num_classes=3)
        model_name = 'googlenet'
    elif args.net == 'transformer':
        model = HistogramViT(
            hist_size=args.hist_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            histograms_per_image=4  # Assuming 4 histograms per image as in the original code
        )
        model_name = 'transformer'
    else:  # Default to ConstancyNetwork
        model = ConstancyNetwork()
        model_name = 'constancy'
    
    # Apply proper weight initialization
    model.apply(init_weights)
    
    criterion = get_loss_function(args.loss)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Set up learning rate scheduler
    if args.scheduler == 'reduce':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        print("Using ReduceLROnPlateau scheduler with factor=0.5, patience=5")
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        print("Using CosineAnnealingWarmRestarts scheduler with T_0=10, T_mult=2, eta_min=1e-6")
    else:
        scheduler = None
        print("No learning rate scheduler used")
    
    # Load checkpoint if restarting training
    start_epoch = 0
    if args.restart_from:
        if os.path.isfile(args.restart_from):
            print(f"Loading checkpoint from {args.restart_from}")
            checkpoint = torch.load(args.restart_from)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Checkpoint file {args.restart_from} not found, starting from scratch")

    # Train the model
    best_val_loss, train_losses, val_losses, epochs_list, learning_rates = train_model(
        model, 
        train_loader,
        val_loader,
        criterion, 
        optimizer,
        scheduler,
        num_epochs=args.epochs, 
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        model_name=f"{model_name}_{args.loss}",
        patience=args.patience,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip
    )

    # Load the best model for evaluation
    best_model_path = os.path.join(args.save_dir, f"{model_name}_{args.loss}_best.pth")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate the model
    print("Evaluating the best model on validation set...")
    metrics, all_errors = evaluate_model(model, val_loader)
    
    # Plot angular error histogram
    plot_angular_error_histogram(all_errors, metrics, args.save_dir, f"{model_name}_{args.loss}")
    
    print("\nEvaluation Results:")
    print(f"Mean Angular Error: {metrics['mean']:.4f}°")
    print(f"Median Angular Error: {metrics['median']:.4f}°")
    print(f"Trimean Angular Error: {metrics['trimean']:.4f}°")
    print(f"Best 25% Angular Error: {metrics['best_25']:.4f}°")
    print(f"Worst 25% Angular Error: {metrics['worst_25']:.4f}°")

    # Save the final evaluation results
    results_path = os.path.join(args.save_dir, f"{model_name}_{args.loss}_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Model: {model_name} with {args.loss} loss\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Mean Angular Error: {metrics['mean']:.4f}°\n")
        f.write(f"Median Angular Error: {metrics['median']:.4f}°\n")
        f.write(f"Trimean Angular Error: {metrics['trimean']:.4f}°\n")
        f.write(f"Best 25% Angular Error: {metrics['best_25']:.4f}°\n")
        f.write(f"Worst 25% Angular Error: {metrics['worst_25']:.4f}°\n")
    
    print(f"Results saved to {results_path}")