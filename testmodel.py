import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from networks.multidataloader import HistogramDataset
from networks.cnn import ConstancyNetwork
from networks.networks import get_loss_function
import argparse
import os
import csv
import datetime
from pathlib import Path

def create_test_dataloader(image_dir, depth_dir, normal_dir, csv_file, batch_size=4, transform=None):
    """Creates a DataLoader for the test dataset."""
    folders = ["folder_test"] 
    test_dataset = HistogramDataset(image_dir, depth_dir, normal_dir, csv_file, folders=folders, plane=True, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader

def collect_outputs(model, dataloader, device):
    """Collects model outputs and targets for histogram generation."""
    model.eval()
    all_outputs = []
    all_targets = []
    
    # Loss functions
    criterion = get_loss_function('euclidean')
    criterion_a = get_loss_function('angular')
    criterion_m = get_loss_function('mse')
    
    total_loss = 0.0
    total_loss_a = 0.0
    total_loss_m = 0.0
    batch_losses = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            # Calculate losses
            loss = criterion(outputs, targets)
            loss_a = criterion_a(outputs, targets)
            loss_m = criterion_m(outputs, targets)
            
            batch_loss = loss.item() * inputs.size(0)
            total_loss += batch_loss
            batch_losses.append(batch_loss / inputs.size(0))
            
            total_loss_a += loss_a.item() * inputs.size(0)
            total_loss_m += loss_m.item() * inputs.size(0)
            
            # Move to CPU for matplotlib processing
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples
    avg_loss_a = total_loss_a / num_samples
    avg_loss_m = total_loss_m / num_samples
    max_batch_loss = max(batch_losses) if batch_losses else 0
    min_batch_loss = min(batch_losses) if batch_losses else 0
    
    metrics = {
        'avg_loss': avg_loss,
        'avg_loss_a': avg_loss_a,
        'avg_loss_m': avg_loss_m,
        'max_batch_loss': max_batch_loss,
        'min_batch_loss': min_batch_loss,
        'num_samples': num_samples
    }
    
    return all_outputs, all_targets, metrics

def create_2d_histograms(outputs, targets, bins=50, save_dir='histograms'):
    """Creates 2D histograms for R-G, G-B, and B-R color pairs."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Define color channel pairs
    channel_pairs = [
        (0, 1, 'R-G'),  # R-G
        (1, 2, 'G-B'),  # G-B
        (2, 0, 'B-R')   # B-R
    ]
    
    # Determine range based on data
    min_val = min(np.min(outputs), np.min(targets))
    max_val = max(np.max(outputs), np.max(targets))
    histogram_range = [[min_val, max_val], [min_val, max_val]]
    
    # Create histograms for each color pair
    for idx1, idx2, name in channel_pairs:
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Target histogram
        h_target, xedges, yedges, im1 = ax1.hist2d(
            targets[:, idx1].flatten(),
            targets[:, idx2].flatten(),
            bins=bins,
            range=histogram_range,
            cmap='viridis',
            norm=plt.cm.colors.LogNorm()
        )
        ax1.set_title(f'Target {name} Histogram')
        ax1.set_xlabel(f'{name[0]} Channel')
        ax1.set_ylabel(f'{name[2]} Channel')
        fig.colorbar(im1, ax=ax1)
        
        # Model output histogram
        h_output, _, _, im2 = ax2.hist2d(
            outputs[:, idx1].flatten(),
            outputs[:, idx2].flatten(),
            bins=bins,
            range=histogram_range,
            cmap='viridis',
            norm=plt.cm.colors.LogNorm()
        )
        ax2.set_title(f'Model Output {name} Histogram')
        ax2.set_xlabel(f'{name[0]} Channel')
        ax2.set_ylabel(f'{name[2]} Channel')
        fig.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'histogram_{name}.png'), dpi=300)
        plt.close(fig)
        
        # Calculate and print histogram difference metrics
        diff = np.sum(np.abs(h_target - h_output)) / np.sum(h_target)
        print(f"{name} Histogram Difference Metric: {diff:.4f}")
        
    # Create a composite figure showing all histograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i, (idx1, idx2, name) in enumerate(channel_pairs):
        # Target histograms (top row)
        h, _, _, im = axes[0, i].hist2d(
            targets[:, idx1].flatten(),
            targets[:, idx2].flatten(),
            bins=bins,
            range=histogram_range,
            cmap='viridis',
            norm=plt.cm.colors.LogNorm()
        )
        axes[0, i].set_title(f'Target {name}')
        fig.colorbar(im, ax=axes[0, i])
        
        # Model output histograms (bottom row)
        h, _, _, im = axes[1, i].hist2d(
            outputs[:, idx1].flatten(),
            outputs[:, idx2].flatten(),
            bins=bins,
            range=histogram_range,
            cmap='viridis',
            norm=plt.cm.colors.LogNorm()
        )
        axes[1, i].set_title(f'Model {name}')
        fig.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'extend_all_histograms.png'), dpi=300)
    plt.close(fig)
    
    print(f"Histograms saved to {save_dir}")

def ensure_csv_exists(csv_path):
    """Ensures the results CSV file exists with the appropriate headers."""
    headers = [
        'timestamp', 'checkpoint', 'transform', 'batch_size', 
        'avg_euclidean_loss', 'avg_angular_loss', 'avg_mse_loss', 
        'max_batch_loss', 'min_batch_loss', 'num_samples'
    ]
    
    if not os.path.exists(csv_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Create CSV with headers
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
    return headers

def save_results_to_csv(csv_path, checkpoint, args, metrics):
    """Saves the test results to the CSV file."""
    headers = ensure_csv_exists(csv_path)
    
    # Extract just the filename from the checkpoint path
    checkpoint_name = os.path.basename(checkpoint)
    
    # Prepare row data
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row_data = {
        'timestamp': timestamp,
        'checkpoint': checkpoint_name,
        'transform': args.transform,
        'batch_size': args.batch_size,
        'avg_euclidean_loss': metrics['avg_loss'],
        'avg_angular_loss': metrics['avg_loss_a'],
        'avg_mse_loss': metrics['avg_loss_m'],
        'max_batch_loss': metrics['max_batch_loss'],
        'min_batch_loss': metrics['min_batch_loss'],
        'num_samples': metrics['num_samples']
    }
    
    # Write to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([row_data[header] for header in headers])
    
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate RGB Histograms')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for data loading')
    parser.add_argument('--transform', type=str, default='log', choices=['tan', 'log', 'None'],
                        help='Transform function for histogram')
    parser.add_argument('--checkpoint', type=str, default='/home/balamurugan.d/src/checkpoints/checkpoints_logRGB_5_extend/best_model.pth', 
                        help='checkpoint')
    parser.add_argument('--bins', type=int, default=64,
                        help='Number of bins for histograms')
    parser.add_argument('--save_dir', type=str, default='histograms',
                        help='Directory to save histograms')
    parser.add_argument('--results_csv', type=str, default='histogram_results.csv',
                        help='Path to save test results CSV file')
    
    args = parser.parse_args()

    print(f"Using {args.checkpoint} model")
    print(f"Using {args.transform} for histogram transform")
    print(f"Using {args.bins} bins for histograms")

    # Ensure we have a valid checkpoint file
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        exit(1)

    # Create full path for results CSV
    results_csv_path = Path(args.results_csv)
    if not results_csv_path.is_absolute():
        # If relative path, create it in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_csv_path = os.path.join(script_dir, args.results_csv)

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = torch.load(args.checkpoint, map_location=device)
        loaded_model = ConstancyNetwork()
        loaded_model.load_state_dict(model['model_state_dict'])
        loaded_model = loaded_model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Create the test dataloader
    test_image_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_training" 
    test_depth_dir = "/work/SuperResolutionData/spectralRatio/data/depth_for_training" 
    test_normal_dir = "/work/SuperResolutionData/spectralRatio/data/surface_norm_for_training" 
    test_csv_file = "/home/balamurugan.d/src/annotations/test_250311_0x.csv" 

    test_dataloader = create_test_dataloader(
        test_image_dir, 
        test_depth_dir, 
        test_normal_dir, 
        test_csv_file,
        batch_size=args.batch_size,
        transform=args.transform
    )

    # Collect outputs and targets
    outputs, targets, metrics = collect_outputs(loaded_model, test_dataloader, device)
    
    print(f"Collected {outputs.shape[0]} samples for histogram generation")
    
    # Print metrics (losses)
    print(f"\nTest Results for model {args.checkpoint}:")
    print(f"Average Euclidean Loss: {metrics['avg_loss']:.4f}")
    print(f"Average Angular Loss: {metrics['avg_loss_a']:.4f}")
    print(f"Average MSE Loss: {metrics['avg_loss_m']:.4f}")
    print(f"Max Batch Loss: {metrics['max_batch_loss']:.4f}")
    print(f"Min Batch Loss: {metrics['min_batch_loss']:.4f}")
    print(f"Number of samples: {metrics['num_samples']}")
    
    # Save results to CSV
    save_results_to_csv(results_csv_path, args.checkpoint, args, metrics)
    
    # Create histograms
    create_2d_histograms(outputs, targets, bins=args.bins, save_dir=args.save_dir)