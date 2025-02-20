import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from datetime import datetime
import math

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Swish activation
class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Sinusoidal positional encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        batch, c, h, w = x.size()
        y_pos = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).repeat(batch, 1, 1, w)
        x_pos = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).repeat(batch, 1, h, 1)

        # Add positional encodings as new channels
        sin_x = torch.sin(math.pi * x_pos)
        cos_x = torch.cos(math.pi * x_pos)
        sin_y = torch.sin(math.pi * y_pos)
        cos_y = torch.cos(math.pi * y_pos)

        pos_encoding = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=1)
        return torch.cat([x, pos_encoding], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.swish = SwishActivation()
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
        out = self.swish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.swish(out)
        return out

class InterferogramNet(nn.Module):
    def __init__(self):
        super(InterferogramNet, self).__init__()
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            SwishActivation()
        )

        # Add positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(32)
        self.post_position = nn.Sequential(
            nn.Conv2d(36, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            SwishActivation()
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

        # Global pooling with both max and average
        self.global_pool_max = nn.AdaptiveMaxPool2d(1)
        self.global_pool_avg = nn.AdaptiveAvgPool2d(1)

        # Separate prediction pathways
        # Tilt parameters (B, C)
        self.tilt_predictor = nn.Sequential(
            nn.Linear(256 * 2, 128),
            SwishActivation(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            SwishActivation(),
            nn.Linear(64, 2)
        )

        # Defocus parameter (D)
        self.defocus_predictor = nn.Sequential(
            nn.Linear(256 * 2, 64),
            SwishActivation(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # Higher order parameters (G, F, J, E, I)
        self.higher_order_predictor = nn.Sequential(
            nn.Linear(256 * 2, 256),
            SwishActivation(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            SwishActivation(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)
        )

        self.apply(self._init_weights)
        self.forward_count = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.6)  # Increased gain
            if m.bias is not None:
                nn.init.uniform_(m.bias, -0.2, 0.2)  # Wider init
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        # Special initialization for final layers to prevent zero predictions
        if isinstance(m, nn.Linear) and m.out_features <= 8:
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -0.3, 0.3)

    def forward(self, x):
        if self.forward_count < 5:
            print(f"\nForward pass {self.forward_count}")
            print(f"Input shape: {x.shape}")
            print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

        # Initial processing with positional encoding
        x = self.initial(x)
        x = self.positional_encoding(x)
        x = self.post_position(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.forward_count < 5:
            print(f"After features shape: {x.shape}")
            print(f"After features range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

        # Global pooling (both max and average)
        x_max = self.global_pool_max(x)
        x_avg = self.global_pool_avg(x)
        pooled = torch.cat([x_max, x_avg], dim=1).view(x.size(0), -1)

        # Separate predictions for different coefficient types
        tilt = self.tilt_predictor(pooled)                # B, C (indices 2, 1)
        defocus = self.defocus_predictor(pooled)          # D (index 0)
        higher_order = self.higher_order_predictor(pooled) # G, F, J, E, I (indices 3, 4, 5, 6, 7)

        # Reorder to match expected output [D, C, B, G, F, J, E, I]
        outputs = torch.cat([
            defocus,                # D (0)
            tilt[:, 1:2],           # C (1)
            tilt[:, 0:1],           # B (2)
            higher_order[:, 0:1],   # G (3)
            higher_order[:, 1:2],   # F (4)
            higher_order[:, 2:3],   # J (5)
            higher_order[:, 3:4],   # E (6)
            higher_order[:, 4:5],   # I (7)
        ], dim=1)

        if self.forward_count < 5:
            print(f"Output shape: {outputs.shape}")
            print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            print(f"Output mean: {outputs.mean():.3f}, std: {outputs.std():.3f}")
            print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

        self.forward_count += 1
        return outputs

class PhaseAwareLoss(nn.Module):
    def __init__(self, similarity_threshold=0.1, diversity_weight=0.1, temperature=1.0, spread_factor=0.3):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.similarity_threshold = similarity_threshold
        self.diversity_weight = diversity_weight
        self.temperature = temperature
        self.spread_factor = spread_factor
        self.epoch = 0

        # Weight coefficients based on their impact on interferograms
        self.coeff_weights = nn.Parameter(
            torch.tensor([1.5, 1.2, 1.2, 0.8, 0.8, 0.8, 0.8, 0.8]),
            requires_grad=False
        )

    def forward(self, pred, target):
        # Ensure inputs are at least 2D (batch_size, num_coeffs)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        # Get coefficients being used
        num_coeffs = pred.shape[1]
        batch_size = pred.shape[0]

        # Use appropriate weights based on which coefficients are active
        if num_coeffs == 1:
            # For single coefficient training, just use weight 1.0
            coeff_weights = torch.ones(1, device=pred.device)
        else:
            # For multiple coefficients, use the preset weights
            # This assumes that the coefficients are ordered consistently
            coeff_weights = self.coeff_weights[:num_coeffs].to(pred.device)

        # Main MSE loss with coefficient weighting
        mse_loss = self.mse(pred, target)
        weighted_mse = (mse_loss * coeff_weights).mean()

        # Identify predictions that are within threshold of target
        close_to_target = torch.abs(pred - target) < 0.2

        # Calculate standard deviation to measure spread
        pred_std = pred.std(dim=0).mean()
        spread_loss = torch.exp(-pred_std / self.spread_factor)

        # Initialize diversity loss with zeros for all predictions
        diversity_penalty = torch.zeros(batch_size, device=pred.device)
        current_weight = 0.0

        if not close_to_target.all():
            # Find predictions that aren't close to target
            # Use flattened mask for indexing if we have multiple coefficients
            if num_coeffs > 1:
                # For multi-coefficient case, consider a prediction "not close"
                # if any of its coefficients are not close
                not_close_mask = ~close_to_target.all(dim=1)

                if not_close_mask.any():
                    # Get predictions where at least one coefficient is not close
                    pred_for_diversity = pred[not_close_mask]

                    if len(pred_for_diversity) > 1:
                        # Calculate pairwise differences between predictions
                        # Use mean across coefficient dimension to get overall similarity
                        scaled_pred = pred_for_diversity / self.temperature

                        # For each pair of predictions, calculate similarity across all coefficients
                        diff_matrix = torch.cdist(scaled_pred, scaled_pred, p=1)
                        too_similar = diff_matrix < (self.similarity_threshold * num_coeffs)
                        too_similar.fill_diagonal_(False)
                        similarity_counts = too_similar.float().sum(dim=1)

                        # Calculate diversity penalty for predictions not close to target
                        not_close_penalty = torch.exp(similarity_counts / batch_size) - 1
                        diversity_penalty[not_close_mask] = not_close_penalty

                        # Weight based on overall similarity
                        current_weight = self.diversity_weight * (1 - torch.exp(-similarity_counts.mean()))
            else:
                # Single coefficient case - simpler handling
                not_close_mask = ~close_to_target.squeeze()

                if not_close_mask.any():
                    pred_for_diversity = pred[not_close_mask].squeeze()

                    if len(pred_for_diversity) > 1:
                        # For single coefficient, direct pairwise differences
                        scaled_pred = pred_for_diversity / self.temperature
                        diffs = scaled_pred.unsqueeze(0) - scaled_pred.unsqueeze(1)
                        too_similar = torch.abs(diffs) < self.similarity_threshold
                        too_similar.fill_diagonal_(False)
                        similarity_counts = too_similar.float().sum(dim=1)

                        not_close_penalty = torch.exp(similarity_counts / batch_size) - 1
                        diversity_penalty[not_close_mask] = not_close_penalty

                        current_weight = self.diversity_weight * (1 - torch.exp(-similarity_counts.mean()))

        # Calculate mean diversity penalty
        diversity_loss = diversity_penalty.mean()

        # Apply epoch-dependent weighting to gradually increase spread importance
        epoch_factor = min(1.0, self.epoch / 20)
        spread_weight = self.spread_factor * epoch_factor

        # Combine all loss components
        total_loss = weighted_mse + (current_weight * diversity_loss) + (spread_weight * spread_loss)

        # Debug prints
        if torch.rand(1).item() < 0.01:  # Print for ~1% of batches
            print(f"\nLoss Components (Epoch {self.epoch}):")
            print(f"Weighted MSE Loss: {weighted_mse:.6f}")
            print(f"Diversity Loss: {diversity_loss:.6f} (weight: {current_weight:.3f})")
            print(f"Spread Loss: {spread_loss:.6f} (weight: {spread_weight:.3f})")
            print(f"Prediction std: {pred_std:.4f}")
            print(f"Total Loss: {total_loss:.6f}")
            print(f"Predictions close to target: {close_to_target.sum().item()}/{batch_size*num_coeffs}")
            print(f"Predictions mean: {pred.mean():.3f}, std: {pred.std():.3f}")

            # Show stats per coefficient if we have multiple
            if num_coeffs > 1:
                for i in range(num_coeffs):
                    coeff_vals = pred[:, i]
                    print(f"  Coeff {i}: mean={coeff_vals.mean():.3f}, std={coeff_vals.std():.3f}")
                    print(f"     range: [{coeff_vals.min():.3f}, {coeff_vals.max():.3f}]")

        return total_loss

    def update_epoch(self, epoch):
        self.epoch = epoch

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
        self.folder_path = folder_path

        print(f"\nLoading dataset from {folder_path}")
        print(f"Active coefficients: {[self.param_names[i] for i in active_coeffs]}")

        # Load all image paths but not the actual images
        self.file_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .jpg files found in {folder_path}")

        print(f"Found {len(self.file_paths)} images")

        # Calculate statistics on a subset of images to save memory
        sample_size = min(1000, len(self.file_paths))
        sample_indices = np.random.choice(len(self.file_paths), sample_size, replace=False)

        sample_params = []
        for idx in sample_indices:
            img_path = self.file_paths[idx]
            # Extract parameters from filename
            params = self._extract_params_from_filename(img_path)
            sample_params.append(params)

        sample_params = torch.stack(sample_params)

        # Print statistics for active parameters
        print("\nParameter statistics (from sample):")
        for i in active_coeffs:
            param_vals = sample_params[:, i].numpy()
            print(f"{self.param_names[i]}: range [{param_vals.min():.3f}, {param_vals.max():.3f}], "
                  f"mean {param_vals.mean():.3f}, std {param_vals.std():.3f}")

    def _extract_params_from_filename(self, img_path):
        """Extract parameters from filename."""
        name = os.path.splitext(os.path.basename(img_path))[0]
        name = name.replace('n', '-').replace('p', '.')
        parts = name.split('_')
        params = np.zeros(8)
        for i, part in enumerate(parts[1:9]):
            params[i] = float(part[1:])
        return torch.FloatTensor(params) / self.cap

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]

        # Load and transform image on demand
        image = Image.open(img_path)
        image = self.transform(image)

        # Extract parameters
        params = self._extract_params_from_filename(img_path)

        # Apply data augmentation with noise to help break symmetry
        if torch.rand(1).item() < 0.3:  # 30% chance of adding noise
            noise = torch.randn_like(image) * 0.03
            image = torch.clamp(image + noise, -1, 1)

        return image, params

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
    target_loss = 0.02  # Loss threshold to move to next phase
    patience_limit = 100  # How many epochs to wait for improvement
    max_epochs_per_phase = 1000
    param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']

    # Initialize model
    model = InterferogramNet().to(device)
    criterion = PhaseAwareLoss(
        similarity_threshold=0.25,
        diversity_weight=1.0,
        temperature=0.8,  # Lower temperature to amplify differences
        spread_factor=0.3
    )

    # Add warm-up phase to help break symmetry
    print("\n=== Starting warm-up phase ===")
    warm_up_epochs = 1

    # Get data loader for warm-up (use everything folder but just one coefficient)
    train_loader, val_loader = create_curriculum_loader('TrainingEverything', [0])

    # Optimizer with higher initial learning rate
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-6)

    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warm_up_epochs, eta_min=1e-5)

    for epoch in range(warm_up_epochs):
        criterion.update_epoch(epoch)
        model.train()
        train_losses = []

        # Add more noise in early epochs
        noise_level = 0.1 * (1 - epoch/warm_up_epochs)

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Add explicit noise to break symmetry
            if noise_level > 0:
                images = images + torch.randn_like(images) * noise_level
                images = torch.clamp(images, -1, 1)

            optimizer.zero_grad()
            outputs = model(images)

            # Add small noise to outputs in early epochs
            if epoch < 2:
                output_noise = torch.randn_like(outputs) * 0.05
                outputs = outputs + output_noise

            # Only compute loss for first coefficient during warm-up
            pred = outputs[:, 0]
            targ = targets[:, 0]
            loss = criterion(pred.unsqueeze(1), targ.unsqueeze(1))

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            if batch_idx % 10 == 0:
                print(f"\rWarm-up Epoch {epoch}: Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {loss.item():.6f} LR: {scheduler.get_last_lr()[0]:.6f}", end='')

        # Validation phase
        model.eval()
        val_losses = []
        val_predictions = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)

                pred = outputs[:, 0]
                targ = targets[:, 0]
                val_loss = criterion(pred.unsqueeze(1), targ.unsqueeze(1))
                val_losses.append(val_loss.item())
                val_predictions.append(pred.cpu())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_predictions = torch.cat(val_predictions)

        print(f"\nWarm-up Epoch {epoch} Summary:")
        print(f"Training Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Prediction Stats - mean: {val_predictions.mean():.3f}, std: {val_predictions.std():.3f}")
        print(f"Prediction Range: [{val_predictions.min():.3f}, {val_predictions.max():.3f}]")

        # Update scheduler
        scheduler.step()

    # Save warm-up model
    torch.save({
        'phase': 'warmup',
        'model_state_dict': model.state_dict(),
    }, os.path.join(log_dir, 'warmup_checkpoint.pth'))

    print(f"\nWarm-up completed. Proceeding with curriculum training.")

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
                criterion.update_epoch(epoch)
                model.train()
                train_losses = []

                # Training phase
                for batch_idx, (images, targets) in enumerate(train_loader):
                    images = images.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)

                    # Only compute loss for active coefficient
                    pred = outputs[:, coeff_idx]
                    targ = targets[:, coeff_idx]
                    loss = criterion(pred.unsqueeze(1), targ.unsqueeze(1))

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

                        pred = outputs[:, coeff_idx]
                        targ = targets[:, coeff_idx]
                        val_loss = criterion(pred.unsqueeze(1), targ.unsqueeze(1))
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
            criterion.update_epoch(epoch)
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