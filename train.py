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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt  # Added for plotting
import numpy as np              # Added for plotting
import random

def add_gaussian_noise(histogram, mean=0, k=0.1):
    """
    Add Gaussian noise to histogram data.
    
    Args:
        histogram: Input histogram tensor
        mean: Mean of the Gaussian noise (default: 0)
        k: Noise scale factor (default: 0.1)
    
    Returns:
        Augmented histogram tensor
    """
    # Calculate standard deviation based on bin count
    bin_count = histogram.shape[-1]  # Assuming last dimension is bin count
    std = k * (bin_count)
    
    # Generate Gaussian noise
    noise = torch.randn_like(histogram) * std + mean
    
    # Add noise to histogram
    augmented_histogram = histogram + noise
    
    # Ensure values remain valid (non-negative for histograms)
    augmented_histogram = torch.clamp(augmented_histogram, min=0.0)
    
    return augmented_histogram



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=10, start_epoch=0, history=None, save_dir='checkpoints', save_interval=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Keep track of best validation loss
    best_val_loss = float('inf')
    
    # Initialize or use provided history
    if history is None:
        history = {
            'train_loss': [],
            'val_loss': []
        }
    
    # If resuming, update best_val_loss from history
    if len(history['val_loss']) > 0:
        best_val_loss = min(history['val_loss'])
        print(f"Best validation loss from previous training: {best_val_loss:.4f}")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        
        # Use tqdm for progress bar
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+num_epochs} [Train]")
        
        for inputs, targets in train_progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            
            # Update progress bar with current loss
            train_progress_bar.set_postfix(loss=loss.item())

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{start_epoch+num_epochs} [Val]")
        
        with torch.no_grad():
            for inputs, targets in val_progress_bar:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item() * inputs.size(0)
                
                # Update progress bar with current loss
                val_progress_bar.set_postfix(loss=loss.item())
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{start_epoch+num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_path = os.path.join(save_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'lr': current_lr,
                'train_loss_history': history['train_loss'],  # Save complete history
                'val_loss_history': history['val_loss'],      # Save complete history
            }, best_model_path)
            print(f"Best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")
        
        # Save model checkpoint at specified intervals
        if (epoch + 1) % save_interval == 0 or epoch == start_epoch + num_epochs - 1:
            plot_training_history(history, save_dir, 'combined_loss', 'adamW')
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'lr': current_lr,
                'train_loss_history': history['train_loss'],  # Save complete history
                'val_loss_history': history['val_loss'],      # Save complete history
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
    print("Training complete!")
    return history
def plot_training_history(history, save_dir, loss_type, optimizer):
    """
    Plot training and validation loss curves and save the plot.
    
    Args:
        history (dict): Dictionary containing train_loss and val_loss lists
        save_dir (str): Directory to save the plot
        loss_type (str): Type of loss function used
        optimizer (str): Type of optimizer used
    """
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-', marker='o', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', marker='s', label='Validation Loss', linewidth=2)
    
    # Find the epoch with the lowest validation loss
    best_epoch = np.argmin(history['val_loss']) + 1
    min_val_loss = min(history['val_loss'])
    
    # Highlight the best model point
    plt.scatter(best_epoch, min_val_loss, s=200, c='green', marker='*', 
                label=f'Best Model (Epoch {best_epoch}, Val Loss: {min_val_loss:.4f})', zorder=3)
    
    plt.title(f'Training and Validation Loss\nLoss: {loss_type}, Optimizer: {optimizer}', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Ensure y-axis starts from 0 if losses are all positive
    if min(history['train_loss'] + history['val_loss']) >= 0:
        plt.ylim(bottom=0)
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f'loss_plot_{loss_type}_{optimizer}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train model with different loss functions')
    parser.add_argument('--loss', type=str, default='combined', choices=['mse', 'angular', 'euclidean', 'combined'],
                        help='Loss function to use: mse, angular, euclidean, or combined')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                        help='Optimizer to use: adam, adamw, sgd, or rmsprop')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save model checkpoints every N epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--transform', type=str, default='None', choices=['tan', 'log', 'None'],
                        help='Transform function for histogram')
    parser.add_argument('--net', type=str, default='None', choices=['google', 'color'],
                        help='Network architecture to use')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--train_csv', type=str, default='/home/balamurugan.d/src/train_250310_10x.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='/home/balamurugan.d/src/val_250310_10x.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    # parser.add_argument('--fourier', type=boo)
    args = parser.parse_args()

    image_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_training"
    depth_dir = "/work/SuperResolutionData/spectralRatio/data/depth_for_training"
    normal_dir = "/work/SuperResolutionData/spectralRatio/data/surface_norm_for_training"
    
    print(f"Using {args.loss} loss function")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    print(f"Using {args.optimizer} optimizer with learning rate: {args.lr}, weight decay: {args.weight_decay}")
    print(f"Scheduler patience: {args.patience}")
    print(f"Saving checkpoints every {args.save_interval} epochs to {args.save_dir}")
    print(f"Training CSV: {args.train_csv}")
    print(f"Validation CSV: {args.val_csv}")

    # Define separate folders for training and validation
    train_folders = [f"folder_{i}" for i in range(1, 10)]
    val_folders = ["folder_val"]  # Use the dedicated validation folder

    # Create separate datasets for training and validation
    train_dataset = HistogramDataset(
        image_dir, depth_dir, normal_dir, 
        args.train_csv, 
        folders=train_folders, 
        plane=True, 
        transform=args.transform,
        training=True,
        fourier_transform=False,
        fourier_filter='lowpass',
    )
    
    val_dataset = HistogramDataset(
        image_dir, depth_dir, normal_dir, 
        args.val_csv, 
        folders=val_folders, 
        plane=True, 
        transform=args.transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # Initialize the model
    if args.net == 'google':
        model = CustomGoogleNet(num_classes=3)
    else:
        model = ConstancyNetwork()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize criterion (loss function)
    criterion = get_loss_function(args.loss)
    
    # Initialize optimizer based on user choice
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)

    # Train the model
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': []}
    
    # Resume from checkpoint if specified
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        # print(checkpoint)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Set starting epoch
        start_epoch = checkpoint['epoch']
        
        # Load history if available
        if 'train_loss' in checkpoint and 'val_loss' in checkpoint:
            history['train_loss'] = checkpoint.get('train_loss_history', [])
            history['val_loss'] = checkpoint.get('val_loss_history', [])
        
        print(f"Resuming from epoch {start_epoch}")
    else:
        if args.resume is not None:
            print(f"Checkpoint {args.resume} not found. Starting training from beginning.")
        else:
            print("Starting training from beginning.")
    
    # Now modify train_model to accept start_epoch and history
    new_history = train_model(
        model, 
        train_loader,
        val_loader, 
        criterion, 
        optimizer,
        scheduler,
        num_epochs=args.epochs,
        start_epoch=start_epoch,  # New parameter
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        history=history  # New parameter
    )

    # Plot and save the training history
    plot_training_history(new_history, args.save_dir, args.loss, args.optimizer)

    # Save the final trained model
    final_model_path = os.path.join(args.save_dir, f"final_model_{args.loss}_{args.optimizer}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
#     python train.py --optimizer adamw --transform log --net color --batch_size 32 --epochs 100 --save_interval 10 --save_dir checkpoints_logRGB_5_extend --resume '/home/balamurugan.d/src/checkpoints_logRGB_5_extend/model_epoch_20.pth'