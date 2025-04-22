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


def train_model(model, dataloader, criterion, optimizer, num_epochs=10, save_dir='checkpoints', save_interval=5, start_epoch=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # model.to(device)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}")
        
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # Update progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Save model checkpoint at specified intervals
        if (epoch + 1) % save_interval == 0 or epoch == start_epoch + num_epochs - 1:
            checkpoint_path = os.path.join(save_dir, f"googlenet_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training complete!")

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
    parser.add_argument('--net', type=str, default='None', choices=['google', 'color'],
                        help='Transform function for histogram')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
# python train_continue.py --loss 'combined' --epoch 110 --save_interval 10 --save_dir 'checkpoints_color_log_combine' --transform 'log' --checkpoint '/home/balamurugan.d/src/checkpoints_color_log_combine/googlenet_epoch_120.pth' --batch_size 32
    image_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_training"
    depth_dir = "/work/SuperResolutionData/spectralRatio/data/depth_for_training"
    normal_dir = "/work/SuperResolutionData/spectralRatio/data/surface_norm_for_training"
    csv_file = "/work/SuperResolutionData/spectralRatio/data/annotation/train_spectral_ratio.csv"

    print(f"Using {args.loss} loss function")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Saving checkpoints every {args.save_interval} epochs to {args.save_dir}")
    folders = [f"folder_{i}" for i in range(1, 10)] 

    dataset = HistogramDataset(image_dir, depth_dir, normal_dir, csv_file, folders=folders, plane=True, transform=args.transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the model based on the network type argument
    if args.net == 'google':
        model = CustomGoogleNet(num_classes=3)
    else:
        model = ConstancyNetwork()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    model.to(device)
    # Get the appropriate loss function
    criterion = get_loss_function(args.loss)
    
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Set the starting epoch
    start_epoch = 0
    
    # Load checkpoint if specified
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Set the starting epoch to the saved epoch
            start_epoch = checkpoint['epoch']
            
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        else:
            print(f"Checkpoint file {args.checkpoint} not found. Starting training from scratch.")

    # Train the model
    train_model(model, dataloader, criterion, optimizer, 
                num_epochs=args.epochs, 
                save_dir=args.save_dir,
                save_interval=args.save_interval,
                start_epoch=start_epoch)

    # Save the final trained model
    final_model_path = os.path.join(args.save_dir, f"Constancy_trained_{args.loss}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")