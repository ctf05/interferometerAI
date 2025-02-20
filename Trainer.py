import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
import numpy as np
import glob
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from datetime import datetime

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.leaky_relu(out)
        return out

class InterferogramNet(nn.Module):
    def __init__(self):
        super(InterferogramNet, self).__init__()

        # Initial convolution with more filters
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Residual blocks with increased capacity
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )

        # Wider classifier without BatchNorm
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 8),
            nn.Tanh()  # Force full range
        )

        self.apply(self._init_weights)
        self.forward_count = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.4)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -0.1, 0.1)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.forward_count < 5:
            print(f"\nForward pass {self.forward_count}")
            print(f"Input shape: {x.shape}")
            print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")

        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.forward_count < 5:
            print(f"After features shape: {x.shape}")
            print(f"After features range: [{x.min():.3f}, {x.max():.3f}]")

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.forward_count < 5:
            print(f"Output shape: {x.shape}")
            print(f"Output range: [{x.min():.3f}, {x.max():.3f}]")

        self.forward_count += 1
        return x

class RangeLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.05):
        super(RangeLoss, self).__init__()
        self.alpha = alpha  # Weight for range penalty
        self.beta = beta    # Weight for L1 loss
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # MSE loss
        mse_loss = self.mse(pred, target)

        # L1 loss for better handling of extremes
        l1_loss = self.beta * self.l1(pred, target)

        # Range penalty - encourage using full range
        range_loss = -self.alpha * (torch.std(pred, dim=0).mean())

        return mse_loss + l1_loss + range_loss

class InterferogramDataset(Dataset):
    def __init__(self, file_paths):
        self.cap = 5.0
        print("Pre-loading all images into RAM...")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(1),
            transforms.Normalize((0.5,), (0.5,))
        ])

        print("\nTransform pipeline:")
        for t in self.transform.transforms:
            print(f"  {t.__class__.__name__}")

        self.images = []
        self.all_params = []

        for i, img_path in enumerate(file_paths):
            if i % 1000 == 0:
                print(f"Loading image {i}/{len(file_paths)}")

            image = Image.open(img_path)
            image = self.transform(image)

            if i < 3:
                print(f"\nImage {i} tensor shape: {image.shape}")
                print(f"Image {i} range: [{image.min():.3f}, {image.max():.3f}]")

            self.images.append(image)

            name = os.path.splitext(os.path.basename(img_path))[0]
            name = name.replace('n', '-').replace('p', '.')
            parts = name.split('_')
            params = np.zeros(8)
            for i, part in enumerate(parts[1:9]):
                params[i] = float(part[1:])

            self.all_params.append(torch.FloatTensor(params) / self.cap)

        self.images = torch.stack(self.images)
        self.all_params = torch.stack(self.all_params)

        print(f"\nFinal dataset statistics:")
        print(f"Images shape: {self.images.shape}")
        print(f"Images range: [{self.images.min():.3f}, {self.images.max():.3f}]")
        print(f"Parameters shape: {self.all_params.shape}")
        print(f"Parameters range: [{self.all_params.min():.3f}, {self.all_params.max():.3f}]")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.all_params[idx]

class DistributionMonitor:
    def __init__(self, log_dir='distribution_plots'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.iteration = 0

    def update(self, predictions, targets):
        if self.iteration % 500 == 0:  # Plot every 500 iterations
            plt.figure(figsize=(15, 5))

            # Plot predictions distribution
            plt.subplot(121)
            plt.hist(predictions.detach().cpu().numpy().flatten(), bins=50, alpha=0.5, label='Predictions')
            plt.hist(targets.detach().cpu().numpy().flatten(), bins=50, alpha=0.5, label='Targets')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.title(f'Distribution at Iteration {self.iteration}')
            plt.legend()

            # Plot range over parameters
            plt.subplot(122)
            pred_ranges = predictions.detach().cpu().abs().mean(0)
            target_ranges = targets.detach().cpu().abs().mean(0)
            params = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']

            plt.bar(np.arange(8) - 0.2, pred_ranges, 0.4, label='Predictions')
            plt.bar(np.arange(8) + 0.2, target_ranges, 0.4, label='Targets')
            plt.xticks(range(8), params)
            plt.ylabel('Mean Absolute Value')
            plt.title('Parameter Ranges')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f'dist_{self.iteration:06d}.png'))
            plt.close()

        self.iteration += 1

def get_data_loaders(root_folder, batch_size=40, num_workers=12):
    print(f"Looking for files in {root_folder}")

    file_paths = glob.glob(os.path.join(root_folder, '*.jpg'))
    print(f"Found {len(file_paths)} files")
    print("Limiting files for testing...")

    np.random.seed(42)
    file_paths = np.random.permutation(file_paths)[:100000].tolist()

    print(f"Actually using {len(file_paths)} files")

    if len(file_paths) == 0:
        raise FileNotFoundError(f"No .jpg files found in {root_folder}")

    print("Creating datasets...")
    full_dataset = InterferogramDataset(file_paths)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader

def train():
    print(f"Running training on {device}")
    start_time = time.time()

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'training_logs_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)

    # Initialize data loaders
    train_loader, val_loader = get_data_loaders('training', batch_size=40, num_workers=12)

    # Initialize model and distribution monitor
    model = InterferogramNet().to(device)
    dist_monitor = DistributionMonitor(os.path.join(log_dir, 'distributions'))

    # Initialize optimizer with larger learning rate
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # Custom scheduler with higher initial LR and slower decay
    num_epochs = 1000
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10
    )

    # Use custom loss that encourages range
    criterion = RangeLoss(alpha=0.1, beta=0.05)

    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 15  # Increased patience
    total_iterations = num_epochs * len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            # Add slight noise to targets during training
            if model.training:
                targets = targets + torch.randn_like(targets) * 0.02

            optimizer.zero_grad()
            outputs = model(images)

            # Update distribution monitor
            dist_monitor.update(outputs, targets)

            loss = criterion(outputs, targets)
            loss.backward()

            # Increased gradient clipping threshold
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            iteration = epoch * len(train_loader) + batch_idx

            if iteration % 10 == 0:
                progress = int(50 * iteration / total_iterations)
                print(f'\rProgress: [{"=" * progress}{" " * (50-progress)}] {iteration}/{total_iterations} | Patience: {patience_counter}', end='')

            if iteration % 1000 == 0:
                model.eval()
                val_loss = 0
                val_predictions = []
                val_targets = []

                with torch.no_grad():
                    for val_images, val_targets_batch in val_loader:
                        val_images = val_images.to(device)
                        val_targets_batch = val_targets_batch.to(device)
                        val_outputs = model(val_images)
                        val_batch_loss = criterion(val_outputs, val_targets_batch)
                        val_loss += val_batch_loss.item()

                        # Store predictions and targets for distribution analysis
                        val_predictions.append(val_outputs.cpu())
                        val_targets.append(val_targets_batch.cpu())

                        del val_images, val_targets_batch, val_outputs, val_batch_loss
                        torch.cuda.empty_cache()

                val_loss /= len(val_loader)

                # Analyze validation predictions
                val_predictions = torch.cat(val_predictions, dim=0)
                val_targets = torch.cat(val_targets, dim=0)

                # Calculate statistics for each parameter
                pred_means = val_predictions.mean(dim=0)
                pred_stds = val_predictions.std(dim=0)
                target_means = val_targets.mean(dim=0)
                target_stds = val_targets.std(dim=0)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save detailed checkpoint
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'iteration': iteration,
                        'pred_stats': {
                            'means': pred_means,
                            'stds': pred_stds
                        }
                    }
                    torch.save(checkpoint, os.path.join(log_dir, 'best_model.pth'))

                    # Save parameter distribution plot
                    plt.figure(figsize=(15, 5))
                    params = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']

                    plt.subplot(121)
                    plt.bar(np.arange(8) - 0.2, pred_means.numpy(), 0.4, yerr=pred_stds.numpy(), label='Predictions')
                    plt.bar(np.arange(8) + 0.2, target_means.numpy(), 0.4, yerr=target_stds.numpy(), label='Targets')
                    plt.xticks(range(8), params)
                    plt.ylabel('Value')
                    plt.title('Parameter Distributions at Best Validation')
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig(os.path.join(log_dir, f'best_distribution_{iteration:06d}.png'))
                    plt.close()
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        print("\nEarly stopping triggered")
                        break

                print(f'\n=== Training Progress at Iteration {iteration} ===')
                print(f'Epoch: {epoch}')
                print(f'Training Loss: {loss.item():.6f}')
                print(f'Validation Loss: {val_loss:.6f}')
                print(f'Parameter Ranges:')
                for i, param in enumerate(params):
                    print(f'{param}: Pred [{pred_means[i]:.3f} ± {pred_stds[i]:.3f}] Target [{target_means[i]:.3f} ± {target_stds[i]:.3f}]')
                print(f'Time Since Start: {(time.time() - start_time)/60:.2f} minutes')
                print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
                print('===================================\n')

                model.train()
                del val_predictions, val_targets
                torch.cuda.empty_cache()

        if patience_counter >= patience_limit:
            break

    # Load best model and test predictions
    checkpoint = torch.load(os.path.join(log_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    print('\n=== Testing Final Predictions ===')
    model.eval()
    with torch.no_grad():
        test_images, actual_params = next(iter(val_loader))
        test_images, actual_params = test_images.to(device), actual_params.to(device)
        predictions = model(test_images)

        # Debug final predictions
        print("\nRaw predictions statistics:")
        print(f"Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"Predictions mean: {predictions.mean():.6f}")
        print(f"Predictions std: {predictions.std():.6f}")

        # Scale back to original range
        predictions = predictions * 5.0
        actual_params = actual_params * 5.0

        for i in range(min(5, predictions.size(0))):
            print(f'\nSample {i+1}:')
            print(f'Predicted: D={predictions[i,0]:.4f}, C={predictions[i,1]:.4f}, '
                  f'B={predictions[i,2]:.4f}, G={predictions[i,3]:.4f}, '
                  f'F={predictions[i,4]:.4f}, J={predictions[i,5]:.4f}, '
                  f'E={predictions[i,6]:.4f}, I={predictions[i,7]:.4f}')

            print(f'Actual:    D={actual_params[i,0]:.4f}, C={actual_params[i,1]:.4f}, '
                  f'B={actual_params[i,2]:.4f}, G={actual_params[i,3]:.4f}, '
                  f'F={actual_params[i,4]:.4f}, J={actual_params[i,5]:.4f}, '
                  f'E={actual_params[i,6]:.4f}, I={actual_params[i,7]:.4f}')

            errors = torch.abs(predictions[i] - actual_params[i])
            print(f'Abs Error: D={errors[0]:.4f}, C={errors[1]:.4f}, '
                  f'B={errors[2]:.4f}, G={errors[3]:.4f}, '
                  f'F={errors[4]:.4f}, J={errors[5]:.4f}, '
                  f'E={errors[6]:.4f}, I={errors[7]:.4f}')

def main():
    train()

if __name__ == '__main__':
    main()