import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
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
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Residual blocks
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

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 8)
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
            print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.forward_count < 5:
            print(f"After features shape: {x.shape}")
            print(f"After features range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.forward_count < 5:
            print(f"Output shape: {x.shape}")
            print(f"Output range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

        self.forward_count += 1
        return x

class DiversityLoss(nn.Module):
    def __init__(self, similarity_threshold=0.1, diversity_weight=0.1, temperature=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.similarity_threshold = similarity_threshold
        self.diversity_weight = diversity_weight
        self.temperature = temperature

    def forward(self, pred, target):
        # Ensure pred and target are 2D tensors [batch_size, num_features]
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        mse_loss = self.mse(pred, target)

        # Scale predictions by temperature to control spreading force
        scaled_pred = pred.view(pred.size(0), -1) / self.temperature  # Flatten to [batch_size, num_features]

        # Compute pairwise differences
        diff_matrix = scaled_pred.unsqueeze(1) - scaled_pred.unsqueeze(0)  # [batch_size, batch_size, num_features]

        # Compute similarity based on L2 norm across feature dimension
        similarity = torch.norm(diff_matrix, dim=2)  # [batch_size, batch_size]
        too_similar = similarity < self.similarity_threshold
        too_similar.fill_diagonal_(False)

        similarity_counts = too_similar.float().sum(dim=1)
        diversity_penalty = torch.exp(similarity_counts / pred.size(0)) - 1
        diversity_loss = diversity_penalty.mean()

        # Adaptive diversity weight
        current_weight = self.diversity_weight * (1 - torch.exp(-similarity_counts.mean()))
        total_loss = mse_loss + current_weight * diversity_loss

        return total_loss

class CurriculumDataset(Dataset):
    def __init__(self, folder_path, active_coeffs):
        """
        Args:
            folder_path: Path to the training folder
            active_coeffs: List of indices of coefficients to train on
        """
        self.cap = 5.0
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(1),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']
        self.active_coeffs = active_coeffs

        print(f"\nLoading dataset from {folder_path}")
        print(f"Active coefficients: {[self.param_names[i] for i in active_coeffs]}")

        # Load all image paths
        file_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        if len(file_paths) == 0:
            raise FileNotFoundError(f"No .jpg files found in {folder_path}")

        # Load images and parameters
        self.images = []
        self.all_params = []

        for i, img_path in enumerate(file_paths):
            if i % 1000 == 0:
                print(f"Loading image {i}/{len(file_paths)}")

            # Extract parameters from filename
            name = os.path.splitext(os.path.basename(img_path))[0]
            name = name.replace('n', '-').replace('p', '.')
            parts = name.split('_')
            params = np.zeros(8)
            for i, part in enumerate(parts[1:9]):
                params[i] = float(part[1:])

            # Load and transform image
            image = Image.open(img_path)
            image = self.transform(image)

            self.images.append(image)
            self.all_params.append(torch.FloatTensor(params) / self.cap)

        self.images = torch.stack(self.images)
        self.all_params = torch.stack(self.all_params)

        # Print statistics for active parameters
        print("\nParameter statistics:")
        for i in active_coeffs:
            param_vals = self.all_params[:, i].numpy()
            print(f"{self.param_names[i]}: range [{param_vals.min():.3f}, {param_vals.max():.3f}], "
                  f"mean {param_vals.mean():.3f}, std {param_vals.std():.3f}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.all_params[idx]

def create_curriculum_loader(folder_path, active_coeffs, batch_size=140, num_workers=64):
    """Create a dataloader for a specific curriculum phase."""
    dataset = CurriculumDataset(folder_path, active_coeffs)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader

def train_curriculum():
    print(f"Starting curriculum training on {device}")
    start_time = time.time()

    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'curriculum_training_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)

    # Training settings
    target_loss = 0.001  # Loss threshold to move to next phase
    patience_limit = 50  # How many epochs to wait for improvement
    max_epochs_per_phase = 100
    param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']

    # Initialize model
    model = InterferogramNet().to(device)
    criterion = DiversityLoss(
        similarity_threshold=0.25,
        diversity_weight=1,
        temperature=1
    )

    # Training phases
    # First, train each coefficient individually with increasing bounds
    for bound in np.arange(0, 5, 0.5):
        print(f"\n=== Starting training phase with bound {bound:.1f} ===")

        for coeff_idx in range(8):
            folder_name = f'Training{param_names[coeff_idx]}{bound:.1f}'.replace('.', '_')
            print(f"\nTraining on {folder_name}")

            # Get data loaders for current phase
            train_loader, val_loader = create_curriculum_loader(folder_name, [coeff_idx])

            # Initialize optimizer and scheduler
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )

            best_val_loss = float('inf')
            patience_counter = 0

            # Training loop for current phase
            for epoch in range(max_epochs_per_phase):
                model.train()
                train_losses = []

                # Training phase
                for batch_idx, (images, targets) in enumerate(train_loader):
                    images = images.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)

                    # Only compute loss for active coefficient
                    loss = criterion(outputs[:, coeff_idx:coeff_idx+1], targets[:, coeff_idx:coeff_idx+1])

                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_losses.append(loss.item())

                    if batch_idx % 10 == 0:
                        print(f"\rEpoch {epoch}: Batch {batch_idx}/{len(train_loader)} "
                              f"Loss: {loss.item():.6f}", end='')

                # Validation phase
                model.eval()
                val_losses = []
                val_predictions = []
                val_targets_list = []

                with torch.no_grad():
                    for images, targets in val_loader:
                        images = images.to(device)
                        targets = targets.to(device)
                        outputs = model(images)

                        val_loss = criterion(outputs[:, coeff_idx:coeff_idx+1],
                                             targets[:, coeff_idx:coeff_idx+1])
                        val_losses.append(val_loss.item())

                        val_predictions.append(outputs[:, coeff_idx].cpu())
                        val_targets_list.append(targets[:, coeff_idx].cpu())

                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)

                # Analyze predictions
                val_predictions = torch.cat(val_predictions)
                val_targets = torch.cat(val_targets_list)

                print(f"\nEpoch {epoch} Summary:")
                print(f"Training Loss: {train_loss:.6f}")
                print(f"Validation Loss: {val_loss:.6f}")
                print(f"Predictions range: [{val_predictions.min():.3f}, {val_predictions.max():.3f}]")
                print(f"Targets range: [{val_targets.min():.3f}, {val_targets.max():.3f}]")
                print(f"Patience: {patience_counter}")

                # Plot prediction distribution
                plt.figure(figsize=(10, 5))
                plt.hist(val_predictions.numpy(), bins=50, alpha=0.5, label='Predictions')
                plt.hist(val_targets.numpy(), bins=50, alpha=0.5, label='Targets')
                plt.title(f'Distribution for {param_names[coeff_idx]} - Epoch {epoch}')
                plt.legend()
                plt.savefig(os.path.join(log_dir, f'{param_names[coeff_idx]}_epoch_{epoch}.png'))
                plt.close()

                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save checkpoint
                    checkpoint_path = os.path.join(
                        log_dir,
                        f'checkpoint_bound{bound:.1f}_{param_names[coeff_idx]}.pth'
                    )
                    torch.save({
                        'bound': bound,
                        'coefficient': coeff_idx,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    print(f"\nSaved checkpoint for {param_names[coeff_idx]}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        print(f"\nEarly stopping for {param_names[coeff_idx]}")
                        break

                # Check if we've reached target loss
                if val_loss < target_loss:
                    print(f"\nReached target loss for {param_names[coeff_idx]}!")
                    break

    # Final phase: TrainingEverything with progressive coefficient activation
    print("\n=== Starting final phase with TrainingEverything ===")

    for num_active_coeffs in range(1, 9):
        active_coeffs = list(range(num_active_coeffs))
        print(f"\nTraining on coefficients: {[param_names[i] for i in active_coeffs]}")

        train_loader, val_loader = create_curriculum_loader('TrainingEverything', active_coeffs)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs_per_phase):
            model.train()
            train_losses = []

            # Training phase
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                # Compute loss for all active coefficients
                loss = criterion(outputs[:, active_coeffs], targets[:, active_coeffs])

                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

                if batch_idx % 10 == 0:
                    print(f"\rEpoch {epoch}: Batch {batch_idx}/{len(train_loader)} "
                          f"Loss: {loss.item():.6f}", end='')

            # Validation phase
            model.eval()
            val_losses = []
            val_predictions = []
            val_targets_list = []

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    outputs = model(images)

                    val_loss = criterion(outputs[:, active_coeffs], targets[:, active_coeffs])
                    val_losses.append(val_loss.item())

                    val_predictions.append(outputs[:, active_coeffs].cpu())
                    val_targets_list.append(targets[:, active_coeffs].cpu())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            # Analyze predictions
            val_predictions = torch.cat(val_predictions)
            val_targets = torch.cat(val_targets_list)

            print(f"\nEpoch {epoch} Summary:")
            print(f"Training Loss: {train_loss:.6f}")
            print(f"Validation Loss: {val_loss:.6f}")

            # Plot prediction distributions for each active coefficient
            plt.figure(figsize=(15, 5 * ((num_active_coeffs + 1) // 2)))
            for i, coeff_idx in enumerate(active_coeffs):
                plt.subplot(((num_active_coeffs + 1) // 2), 2, i + 1)
                plt.hist(val_predictions[:, i].numpy(), bins=50, alpha=0.5, label='Predictions')
                plt.hist(val_targets[:, i].numpy(), bins=50, alpha=0.5, label='Targets')
                plt.title(f'Distribution for {param_names[coeff_idx]}')
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f'final_phase_epoch_{epoch}.png'))
            plt.close()

            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save checkpoint
                checkpoint_path = os.path.join(
                    log_dir,
                    f'final_checkpoint_coeffs_{num_active_coeffs}.pth'
                )
                torch.save({
                    'num_active_coeffs': num_active_coeffs,
                    'active_coeffs': active_coeffs,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"\nSaved checkpoint for {num_active_coeffs} coefficients")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"\nEarly stopping for {num_active_coeffs} coefficients")
                    break

            # Check if we've reached target loss
            if val_loss < target_loss:
                print(f"\nReached target loss for {num_active_coeffs} coefficients!")
                break

    print("\nCurriculum training completed!")
    print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_time': time.time() - start_time,
    }, os.path.join(log_dir, 'final_model.pth'))

    return model

def test_model(model, test_images):
    """Test the model on a batch of images."""
    model.eval()
    with torch.no_grad():
        predictions = model(test_images.to(device))
        # Scale predictions back to original range
        predictions = predictions * 5.0
        return predictions

def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Train the model using curriculum learning
    model = train_curriculum()
    print("\nTraining completed. Model saved.")

if __name__ == '__main__':
    main()