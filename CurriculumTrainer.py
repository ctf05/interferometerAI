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

        # Classifier without final activation
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

class DiversityLoss(nn.Module):
    def __init__(self, similarity_threshold=0.1, diversity_weight=0.1, temperature=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.similarity_threshold = similarity_threshold
        self.diversity_weight = diversity_weight
        self.temperature = temperature

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)

        # Scale predictions by temperature to control spreading force
        scaled_pred = pred / self.temperature

        diff_matrix = scaled_pred.unsqueeze(0) - scaled_pred.unsqueeze(1)
        too_similar = torch.abs(diff_matrix) < self.similarity_threshold
        too_similar.fill_diagonal_(False)

        similarity_counts = too_similar.float().sum(dim=1)
        diversity_penalty = torch.exp(similarity_counts / pred.shape[0]) - 1
        diversity_loss = diversity_penalty.mean()

        # Could also use curriculum on the diversity weight
        current_weight = self.diversity_weight * (1 - torch.exp(-similarity_counts.mean()))

        total_loss = mse_loss + current_weight * diversity_loss

        return total_loss

class CurriculumDataset(Dataset):
    def __init__(self, file_paths, active_coeff=0, restricted_range=2.5):
        """
        Args:
            file_paths: List of image file paths
            active_coeff: Index of the coefficient to train on (0-7)
            restricted_range: Range limit for inactive coefficients [-restricted_range, restricted_range]
        """
        self.cap = 5.0
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(1),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Parameter names for logging
        self.param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']

        print(f"\nCreating curriculum dataset focusing on {self.param_names[active_coeff]}")
        print(f"Filtering {len(file_paths)} initial files...")

        # Filter files based on coefficient ranges
        filtered_paths = []
        filtered_params = []

        for img_path in file_paths:
            name = os.path.splitext(os.path.basename(img_path))[0]
            name = name.replace('n', '-').replace('p', '.')
            parts = name.split('_')
            params = np.zeros(8)
            for i, part in enumerate(parts[1:9]):
                params[i] = float(part[1:])

            # Check if inactive coefficients are within restricted range
            # and active coefficient uses full range
            is_valid = True
            for i in range(8):
                if i == active_coeff:
                    if abs(params[i]) <= restricted_range / 3:
                        is_valid = False
                else:
                    if abs(params[i]) > restricted_range:
                        is_valid = False

            if is_valid:
                filtered_paths.append(img_path)
                filtered_params.append(params)

                if len(filtered_paths) >= 10000:  # Stop once we have enough
                    break

        print(f"Found {len(filtered_paths)} valid files")
        if len(filtered_paths) < 1000:
            raise ValueError(f"Not enough valid files found! Only got {len(filtered_paths)}")

        # Load images and parameters
        self.images = []
        self.all_params = []

        for i, (img_path, params) in enumerate(zip(filtered_paths, filtered_params)):
            if i % 1000 == 0:
                print(f"Loading image {i}/{len(filtered_paths)}")

            image = Image.open(img_path)
            image = self.transform(image)
            self.images.append(image)
            self.all_params.append(torch.FloatTensor(params) / self.cap)

        self.images = torch.stack(self.images)
        self.all_params = torch.stack(self.all_params)

        # Print statistics for each parameter
        print("\nParameter statistics:")
        for i in range(8):
            param_vals = self.all_params[:, i].numpy()
            print(f"{self.param_names[i]}: range [{param_vals.min():.3f}, {param_vals.max():.3f}], "
                  f"mean {param_vals.mean():.3f}, std {param_vals.std():.3f}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.all_params[idx]

def create_curriculum_loader(root_folder, active_coeff, batch_size=40, num_workers=12):
    """
    Create a dataloader for a specific curriculum phase.
    """
    print(f"\nCreating curriculum dataloader for coefficient {active_coeff}")
    file_paths = glob.glob(os.path.join(root_folder, '*.jpg'))

    if len(file_paths) == 0:
        raise FileNotFoundError(f"No .jpg files found in {root_folder}")

    # Create dataset with filtered images
    dataset = CurriculumDataset(file_paths, active_coeff)

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
    target_loss = 0.001  # Loss threshold to move to next coefficient
    patience_limit = 50  # How many epochs to wait for improvement
    max_epochs_per_coeff = 100  # Maximum epochs per coefficient

    # Initialize model
    model = InterferogramNet().to(device)
    param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']

    # Training loop for each coefficient
    for active_coeff in range(8):
        print(f"\n=== Starting training for coefficient {param_names[active_coeff]} ===")

        # Get curriculum-specific data loaders
        train_loader, val_loader = create_curriculum_loader('training', active_coeff)

        # Initialize optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = DiversityLoss(
            similarity_threshold=.25,
            diversity_weight=1,
            temperature=1
        )

        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop for current coefficient
        for epoch in range(max_epochs_per_coeff):
            model.train()
            train_losses = []

            # Training phase
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                # Only compute loss for active coefficient
                pred = outputs[:, active_coeff]
                targ = targets[:, active_coeff]
                loss = criterion(pred, targ)
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
                    pred = outputs[:, active_coeff]
                    targ = targets[:, active_coeff]
                    val_loss = criterion(pred, targ)
                    val_losses.append(val_loss.item())

                    # Store predictions and targets for analysis
                    val_predictions.append(outputs[:, active_coeff].cpu())
                    val_targets_list.append(targets[:, active_coeff].cpu())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            # Analyze predictions
            val_predictions = torch.cat(val_predictions)
            val_targets = torch.cat(val_targets_list)

            print(f"\nEpoch {epoch} Summary:")
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
            plt.title(f'Distribution for {param_names[active_coeff]} - Epoch {epoch}')
            plt.legend()
            plt.savefig(os.path.join(log_dir, f'{param_names[active_coeff]}_epoch_{epoch}.png'))
            plt.close()

            scheduler.step(val_loss)

            # Check if we should move to next coefficient
            if val_loss < target_loss:
                print(f"\nReached target loss for {param_names[active_coeff]}!")
                break

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save checkpoint
                checkpoint_path = os.path.join(log_dir, f'curriculum_checkpoint_{param_names[active_coeff]}.pth')
                torch.save({
                    'coefficient': active_coeff,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)

                print(f"\nSaved checkpoint for {param_names[active_coeff]}")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"\nEarly stopping for {param_names[active_coeff]}")
                    break

        print(f"\nCompleted training for {param_names[active_coeff]}")
        print(f"Best validation loss: {best_val_loss:.6f}")

        # Final evaluation for this coefficient
        model.eval()
        with torch.no_grad():
            all_predictions = []
            all_targets = []

            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)

                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Create final plots for this coefficient
            plt.figure(figsize=(15, 5))

            # Distribution plot
            plt.subplot(121)
            plt.hist(all_predictions[:, active_coeff].numpy(), bins=50, alpha=0.5, label='Predictions')
            plt.hist(all_targets[:, active_coeff].numpy(), bins=50, alpha=0.5, label='Targets')
            plt.title(f'Final Distribution for {param_names[active_coeff]}')
            plt.legend()

            # Scatter plot
            plt.subplot(122)
            plt.scatter(all_targets[:, active_coeff].numpy(),
                        all_predictions[:, active_coeff].numpy(),
                        alpha=0.1)
            plt.plot([-1, 1], [-1, 1], 'r--')  # Perfect prediction line
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'Prediction Accuracy for {param_names[active_coeff]}')

            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f'{param_names[active_coeff]}_final_analysis.png'))
            plt.close()

    print("\nCurriculum training completed!")
    print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_time': time.time() - start_time,
    }, os.path.join(log_dir, 'final_model.pth'))

    return model

def test_model(model, test_images):
    """
    Test the model on a batch of images.
    """
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