import torch
from torch.utils.data import DataLoader
from multidataloader import HistogramDataset
from cnn import ConstancyNetwork
from networks import CustomGoogleNet, get_loss_function
import argparse
import os
import csv
import datetime
from pathlib import Path

def create_test_dataloader(image_dir, depth_dir, normal_dir, csv_file, batch_size=4, transform=None):
    """Creates a DataLoader for the test dataset."""
    folders = [f"folder_test"] 
    test_dataset = HistogramDataset(image_dir, depth_dir, normal_dir, csv_file, folders=folders, plane=True, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader

def test_model(model, dataloader, device, loss_type):
    """Tests the model and returns the average loss and additional metrics."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_loss_a = 0.0
    total_loss_m = 0.0
    criterion = get_loss_function('euclidean')
    criterion_a = get_loss_function('angular')
    criterion_m = get_loss_function('mse')
    
    # Additional metrics
    batch_losses = []
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_a = criterion_a(outputs, targets)
            loss_m = criterion_m(outputs, targets)
            batch_loss = loss.item() * inputs.size(0)
            total_loss += batch_loss
            batch_losses.append(batch_loss / inputs.size(0))
            batch_loss_a = loss_a.item() * inputs.size(0)
            total_loss_a += batch_loss_a
            batch_loss_m = loss_m.item() * inputs.size(0)
            total_loss_m += batch_loss_m

    avg_loss = total_loss / len(dataloader.dataset)
    avg_loss_a = total_loss_a / len(dataloader.dataset)
    avg_loss_m = total_loss_m / len(dataloader.dataset)
    max_batch_loss = max(batch_losses) if batch_losses else 0
    min_batch_loss = min(batch_losses) if batch_losses else 0
    
    metrics = {
        'avg_loss':avg_loss,
        'avg_loss_a': avg_loss_a,
        'avg_loss_m': avg_loss_m,
        'max_batch_loss': max_batch_loss,
        'min_batch_loss': min_batch_loss,
        'num_samples': len(dataloader.dataset)
    }
    
    return metrics

def ensure_csv_exists(csv_path):
    """Ensures the results CSV file exists with the appropriate headers."""
    headers = [
        'timestamp', 'checkpoint', 'transform', 'batch_size',
        'avg_euclidean_loss', 'avg_angular_loss', 'avg_mse_loss', 'max_batch_loss', 'min_batch_loss', 'num_samples'
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
        'checkpoint': checkpoint,
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
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('--loss', type=str, default='euclidean', choices=['mse', 'angular', 'euclidean'],
                        help='Loss function to use: mse, angular, or euclidean')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--transform', type=str, default='log', choices=['tan', 'log', 'None'],
                        help='Transform function for histogram')
    parser.add_argument('--checkpoint', type=str, default='/home/balamurugan.d/src/checkpoints_logRGB_5_extend/best_model.pth', 
                        help='checkpoint')
    parser.add_argument('--results_csv', type=str, default='test_results.csv',
                        help='Path to save test results CSV file')
    
    args = parser.parse_args()

    print(f"Using {args.loss} loss function")
    print(f"Using {args.checkpoint} model")
    print(f"Using {args.transform} for histogram transform")

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
    try:
        model = torch.load(args.checkpoint,map_location=device)
        loaded_model = ConstancyNetwork()
        # loaded_model = CustomGoogleNet()
        loaded_model.load_state_dict(model['model_state_dict'])
        # loaded_model.load_state_dict(model)

    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    loaded_model = loaded_model.to(device)

    # Create the test dataloader
    test_image_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_training" 
    test_depth_dir = "/work/SuperResolutionData/spectralRatio/data/depth_for_training" 
    test_normal_dir = "/work/SuperResolutionData/spectralRatio/data/surface_norm_for_training" 
    test_csv_file = "/home/balamurugan.d/src/test.csv" 

    test_dataloader = create_test_dataloader(
        test_image_dir, 
        test_depth_dir, 
        test_normal_dir, 
        test_csv_file,
        batch_size=args.batch_size,
        transform=args.transform
    )

    # Test the model
    test_metrics = test_model(loaded_model, test_dataloader, device, args.loss)
    
    # Print results
    print(f"Test Results for model {args.checkpoint}:")
    print(f"Average Loss: {test_metrics['avg_loss']:.4f}")
    print(f"Max Batch Loss: {test_metrics['max_batch_loss']:.4f}")
    print(f"Min Batch Loss: {test_metrics['min_batch_loss']:.4f}")
    print(f"Number of samples: {test_metrics['num_samples']}")
    
    # Save results to CSV
    save_results_to_csv(results_csv_path, args.checkpoint, args, test_metrics)