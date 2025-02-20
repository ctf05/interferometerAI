import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from datetime import datetime
import math
import copy

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Swish activation
class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Sinusoidal positional encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        batch, c, h, w = x.size()

        # Generate normalized coordinate grids
        y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).repeat(batch, 1, 1, w)
        x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).repeat(batch, 1, h, 1)

        # Compute radial distance from center (important for circular patterns)
        r = torch.sqrt(x_coords**2 + y_coords**2)

        # Calculate angular position (theta)
        theta = torch.atan2(y_coords, x_coords)

        # Create positional encodings
        pos_enc = []

        # Radial encodings at different frequencies
        for freq in [1, 2, 4, 8]:
            pos_enc.append(torch.sin(r * math.pi * freq))
            pos_enc.append(torch.cos(r * math.pi * freq))

        # Angular encodings at different frequencies
        for freq in [1, 2, 4]:
            pos_enc.append(torch.sin(theta * freq))
            pos_enc.append(torch.cos(theta * freq))

        # Cartesian encodings
        pos_enc.append(torch.sin(x_coords * math.pi))
        pos_enc.append(torch.cos(x_coords * math.pi))
        pos_enc.append(torch.sin(y_coords * math.pi))
        pos_enc.append(torch.cos(y_coords * math.pi))

        # Concatenate all encodings
        pos_encoding = torch.cat(pos_enc, dim=1)  # Should be 16 channels

        # Concatenate with input
        return torch.cat([x, pos_encoding], dim=1)

class FourierConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Real and imaginary convolutions in frequency domain
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Apply 2D FFT
        x_fft = torch.fft.rfft2(x)

        # Get real and imaginary components
        real, imag = x_fft.real, x_fft.imag

        # Apply convolutions in frequency domain with stride
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)

        # Recombine and apply inverse FFT
        x_fft_out = torch.complex(real_out, imag_out)

        # Compute output size based on stride
        output_size = (x.shape[2] // self.stride, x.shape[3] // self.stride)
        x_out = torch.fft.irfft2(x_fft_out, s=output_size)

        return self.bn(x_out)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_fourier=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Optional Fourier branch
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierConvLayer(in_channels, out_channels, stride=stride)
            self.fourier_weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        # Regular convolution path
        residual = x
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Add Fourier path if enabled
        if self.use_fourier:
            fourier_out = self.fourier(x)
            out = out + self.fourier_weight * fourier_out

        out += self.shortcut(residual)
        out = self.leaky_relu(out)
        return out

class InterferogramNet(nn.Module):
    def __init__(self):
        super(InterferogramNet, self).__init__()

        # 1. Initial convolution with coordinate-aware features
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.positional_encoding = SinusoidalPositionalEncoding(32)

        # Combine and adjust channels after positional encoding
        # Input becomes 32 + 16 = 48 channels
        self.post_positional = nn.Sequential(
            nn.Conv2d(50, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 2. Residual blocks with Fourier processing
        # First layer doesn't use Fourier to save computation
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, use_fourier=False),
            ResidualBlock(64, 64, use_fourier=True)
        )
        # Middle and later layers use Fourier processing
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_fourier=True),
            ResidualBlock(128, 128, use_fourier=True)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_fourier=True),
            ResidualBlock(256, 256, use_fourier=True)
        )

        # Global pooling (both max and average for better feature representation)
        self.global_pool_avg = nn.AdaptiveAvgPool2d(1)
        self.global_pool_max = nn.AdaptiveMaxPool2d(1)

        # 3. Separate predictors for different coefficient types
        # Low-order coefficients (Defocus: D)
        self.defocus_predictor = nn.Sequential(
            nn.Linear(512, 128),  # 512 = 256*2 (from avg+max pooling)
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)  # Predict D coefficient
        )

        # Tilt coefficients (B, C)
        self.tilt_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2)  # Predict B, C coefficients
        )

        # Astigmatism coefficients (E, I)
        self.astigmatism_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2)  # Predict E, I coefficients
        )

        # Higher-order coefficients (G, F, J)
        self.higher_order_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 3)  # Predict G, F, J coefficients
        )

        self.apply(self._init_weights)
        self.forward_count = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.4)
            if m.bias is not None:
                # Use non-zero bias initialization to help break symmetry
                nn.init.uniform_(m.bias, -0.1, 0.1)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        # Special initialization for the final layer of each predictor
        # This helps prevent the "all zeros" prediction problem
        if isinstance(m, nn.Linear) and hasattr(m, 'out_features'):
            if m.out_features <= 3:  # Final layer of any predictor
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.2, 0.2)

    def forward(self, x):
        if self.forward_count < 5:
            print(f"\nForward pass {self.forward_count}")
            print(f"Input shape: {x.shape}")
            print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
            if torch.cuda.is_available():
                print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

        # Initial processing with positional encoding
        x = self.initial(x)
        x = self.positional_encoding(x)  # Adds coordinate-aware features
        x = self.post_positional(x)

        # Feature extraction through residual blocks with Fourier processing
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.forward_count < 5:
            print(f"After features shape: {x.shape}")
            print(f"After features range: [{x.min():.3f}, {x.max():.3f}]")
            if torch.cuda.is_available():
                print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

        # Global pooling (both average and max for better features)
        x_avg = self.global_pool_avg(x).view(x.size(0), -1)
        x_max = self.global_pool_max(x).view(x.size(0), -1)
        x_pooled = torch.cat([x_avg, x_max], dim=1)

        # Separate predictions through specialized pathways
        defocus_out = self.defocus_predictor(x_pooled)            # D (index 0)
        tilt_out = self.tilt_predictor(x_pooled)                  # B, C (indices 2, 1)
        astigmatism_out = self.astigmatism_predictor(x_pooled)    # E, I (indices 6, 7)
        higher_out = self.higher_order_predictor(x_pooled)        # G, F, J (indices 3, 4, 5)

        # Reorder outputs to match expected [D, C, B, G, F, J, E, I] format
        outputs = torch.cat([
            defocus_out,                  # D (0)
            tilt_out[:, 1:2],             # C (1)
            tilt_out[:, 0:1],             # B (2)
            higher_out[:, 0:1],           # G (3)
            higher_out[:, 1:2],           # F (4)
            higher_out[:, 2:3],           # J (5)
            astigmatism_out[:, 0:1],      # E (6)
            astigmatism_out[:, 1:2],      # I (7)
        ], dim=1)

        if self.forward_count < 5:
            print(f"Output shape: {outputs.shape}")
            print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

        self.forward_count += 1
        return outputs

class PhaseAwareLoss(nn.Module):
    def __init__(self, similarity_threshold=0.1, diversity_weight=0.1, temperature=1.0, spread_factor=0.3, warm_up_epochs=5):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.similarity_threshold = similarity_threshold
        self.diversity_weight = diversity_weight
        self.temperature = temperature
        self.spread_factor = spread_factor
        self.epoch = 0
        self.warm_up_epochs = warm_up_epochs

        # Weight coefficients based on their impact on interferograms
        self.coeff_weights = nn.Parameter(
            torch.tensor([1.5, 1.2, 1.2, 0.8, 0.8, 0.8, 0.8, 0.8]),
            requires_grad=False
        )

    def saturated_loss(self, pred, target):
        # Calculate absolute difference
        diff = torch.abs(pred - target)

        # Clamp difference at 0.2 (maintain original saturation)
        saturated_diff = torch.clamp(diff, max=0.2)

        # Apply edge correction with smooth quadratic profile
        # This avoids creating local minima in the expected loss
        edge_correction = torch.zeros_like(pred)

        # Calculate normalized distance from center
        center_distance = torch.abs(pred)

        # Apply smooth quadratic correction in edge regions
        in_edge_region = (center_distance >= 0.8) & (center_distance <= 1.0)

        # Use quadratic function for smooth transition
        # This creates a smooth bowl-shaped correction that exactly
        # compensates for the edge disadvantage
        edge_position = (center_distance[in_edge_region] - 0.8) / 0.2  # 0 to 1 scale

        # Quadratic profile: a*x² matches edge disadvantage of 0.00990 at x=1
        correction_amount = -0.00990 * edge_position**2
        edge_correction[in_edge_region] = correction_amount

        # Add out-of-range penalty
        out_of_range = torch.clamp(torch.abs(pred) - 1.0, min=0)
        range_penalty = out_of_range ** 2

        # Combine all components
        balanced_loss = saturated_diff + edge_correction + range_penalty

        return balanced_loss

    def forward(self, pred, target, is_training=False, warm_up_epochs=5):
        self.warm_up_epochs = warm_up_epochs
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

        # Main loss with coefficient weighting
        base_loss = self.saturated_loss(pred, target)
        weighted_base_loss = (base_loss * coeff_weights).mean()

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
        total_loss = weighted_base_loss + (current_weight * diversity_loss) + (spread_weight * spread_loss)

        # Debug prints
        if torch.rand(1).item() < 0.01:  # Print for ~1% of batches
            print(f"\nLoss Components (Epoch {self.epoch}):")
            print(f"Weighted Base Loss: {weighted_base_loss:.6f}")
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

        if self.epoch < self.warm_up_epochs and is_training:
            min_prediction_magnitude = 0.8 * (1 - self.epoch / self.warm_up_epochs)
            # Lower bound penalty (force away from zero)
            outside_zero_penalty = torch.clamp(min_prediction_magnitude - torch.abs(pred), min=0)
            # Upper bound penalty (force within [-1,1])
            outside_range_penalty = torch.clamp(torch.abs(pred) - 1.0, min=0)

            # Range coverage penalty - check if we're using both positive and negative values
            max_vals = torch.max(pred, dim=0)[0]
            min_vals = torch.min(pred, dim=0)[0]
            range_coverage_penalty = torch.clamp(0.5 - max_vals, min=0) + torch.clamp(min_vals + 0.5, min=0)

            # Combine all penalties
            zero_constraint_loss = 10 * (outside_zero_penalty + outside_range_penalty) + 5 * range_coverage_penalty
            total_loss = zero_constraint_loss.mean() * 100

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

def get_cyclic_lr(epoch, batch_idx, batches_per_epoch, base_lr=1e-4, max_lr=1e-3):
    cycle_length = 10  # epochs
    cycle = (epoch + batch_idx/batches_per_epoch) / cycle_length
    cycle_position = cycle - int(cycle)

    if cycle_position < 0.5:
        # Increasing phase
        return base_lr + (max_lr - base_lr) * (2 * cycle_position)
    else:
        # Decreasing phase
        return max_lr - (max_lr - base_lr) * (2 * (cycle_position - 0.5))

def train_with_genetic_exploration(model, train_loader, criterion, population_size=5, batch_samples=5, device=device):
    """
    Implement genetic-style exploration to help escape local minima.
    Returns the best performing model from the population.
    """
    print(f"\nRunning genetic exploration with population size {population_size}")

    # Create a population of models with variations
    population = [copy.deepcopy(model) for _ in range(population_size)]

    # Add parameter noise to each model (except the first one which stays as-is)
    for i in range(1, population_size):
        noise_scale = 0.02 * (i / (population_size - 1) + 0.5)  # Increasing noise levels
        for param in population[i].parameters():
            noise = torch.randn_like(param) * noise_scale
            param.data += noise
        print(f"Added {noise_scale:.3f} scale noise to model variant {i}")

    # Train each model for a few batches and track fitness
    fitness_scores = []
    val_predictions = []

    for model_idx, model_variant in enumerate(population):
        model_variant.to(device)
        model_variant.train()
        optimizer = optim.AdamW(model_variant.parameters(), lr=1e-4)

        losses = []
        all_preds = []

        # Train on a few batches
        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx >= batch_samples:
                break

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model_variant(images)

            loss = criterion(outputs, targets)
            loss.backward()
            clip_grad_norm_(model_variant.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            all_preds.append(outputs.detach().cpu())

        # Calculate fitness score based on:
        # 1. Loss value (lower is better)
        # 2. Prediction spread (higher std is better)
        avg_loss = np.mean(losses)
        all_preds = torch.cat(all_preds)
        pred_std = all_preds.std().item()

        # Combined fitness score: negative loss (higher is better) plus spread bonus
        fitness = -avg_loss + 0.2 * pred_std
        fitness_scores.append(fitness)
        val_predictions.append(all_preds)

        print(f"Model {model_idx}: Loss={avg_loss:.4f}, Std={pred_std:.4f}, Fitness={fitness:.4f}")

    # Select best model
    best_idx = np.argmax(fitness_scores)
    best_model = population[best_idx]
    best_fitness = fitness_scores[best_idx]

    # Plot predictions distributions from different population members
    plt.figure(figsize=(12, 8))
    for i, preds in enumerate(val_predictions):
        plt.subplot(3, 2, i+1)
        plt.hist(preds.numpy().flatten(), bins=30)
        plt.title(f"Model {i}: Fitness={fitness_scores[i]:.4f}")
    plt.tight_layout()
    plt.savefig(f"genetic_exploration_distributions_{int(time.time())}.png")
    plt.close()

    # Optional: Perform crossover between top models
    if population_size >= 3:
        # Sort indices by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        parent1_idx = sorted_indices[0]
        parent2_idx = sorted_indices[1]

        # Create child through crossover of top two models
        child_model = copy.deepcopy(population[parent1_idx])
        parent2 = population[parent2_idx]

        # Simple crossover: randomly select parameters from either parent
        with torch.no_grad():
            for child_param, parent2_param in zip(child_model.parameters(), parent2.parameters()):
                # Randomly choose whether to use parent1 (keep as is) or parent2 param
                mask = torch.rand_like(child_param) > 0.5
                child_param.data[mask] = parent2_param.data[mask]

        # Evaluate the child
        child_model.to(device)
        child_model.train()
        child_optimizer = optim.AdamW(child_model.parameters(), lr=1e-4)

        child_losses = []
        child_preds = []

        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx >= batch_samples:
                break

            images = images.to(device)
            targets = targets.to(device)

            child_optimizer.zero_grad()
            outputs = child_model(images)

            loss = criterion(outputs, targets)
            loss.backward()
            clip_grad_norm_(child_model.parameters(), max_norm=1.0)
            child_optimizer.step()

            child_losses.append(loss.item())
            child_preds.append(outputs.detach().cpu())

        # Evaluate child fitness
        avg_child_loss = np.mean(child_losses)
        all_child_preds = torch.cat(child_preds)
        child_pred_std = all_child_preds.std().item()
        child_fitness = -avg_child_loss + 0.2 * child_pred_std

        print(f"Child model: Loss={avg_child_loss:.4f}, Std={child_pred_std:.4f}, Fitness={child_fitness:.4f}")

        # Use child if it's better than the best parent
        if child_fitness > best_fitness:
            best_model = child_model
            best_fitness = child_fitness
            print("Child model outperforms best parent and will be used!")

    # Return the best model
    best_model.to(device)
    return best_model

def train_curriculum():
    print(f"Starting curriculum training on {device}")
    start_time = time.time()

    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'curriculum_training_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)

    # Training settings
    target_loss = 0.015  # Adjusted target loss to be more realistic
    patience_limit = 100  # How many epochs to wait for improvement
    max_epochs_per_phase = 1000
    param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']

    # Genetic exploration settings
    genetic_exploration_frequency = 10  # Run genetic exploration every N epochs
    population_size = 5

    # Initialize model
    model = InterferogramNet().to(device)
    criterion = PhaseAwareLoss(
        similarity_threshold=0.25,
        diversity_weight=1.0,
        temperature=0.8,
        spread_factor=0.3
    )

    # Add warm-up phase to help break symmetry
    print("\n=== Starting warm-up phase ===")
    warm_up_epochs = 5

    # Get data loader for warm-up (use everything folder but just one coefficient)
    train_loader, val_loader = create_curriculum_loader('TrainingD0_0', [0])
    batches_per_epoch = len(train_loader)

    # Use cyclic learning rates instead of fixed
    base_lr, max_lr = 5e-5, 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4, eps=1e-6)

    for epoch in range(warm_up_epochs):
        criterion.update_epoch(epoch)
        model.train()
        train_losses = []
        lr_values = []

        # Add more noise in early epochs
        noise_level = 0.1 * (1 - epoch/warm_up_epochs)

        for batch_idx, (images, targets) in enumerate(train_loader):
            # Update learning rate for this batch
            current_lr = get_cyclic_lr(epoch, batch_idx, batches_per_epoch, base_lr, max_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            lr_values.append(current_lr)

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
            loss = criterion(pred.unsqueeze(1), targ.unsqueeze(1), is_training=True, warm_up_epochs=warm_up_epochs)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            if batch_idx % 10 == 0:
                print(f"\rWarm-up Epoch {epoch}: Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {loss.item():.6f} LR: {current_lr:.6f}", end='')

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
        avg_lr = np.mean(lr_values)

        print(f"\nWarm-up Epoch {epoch} Summary:")
        print(f"Training Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Average LR: {avg_lr:.6f}")
        print(f"Prediction Stats - mean: {val_predictions.mean():.3f}, std: {val_predictions.std():.3f}")
        print(f"Prediction Range: [{val_predictions.min():.3f}, {val_predictions.max():.3f}]")

        # Plot warm-up distribution and learning rate
        plt.figure(figsize=(12, 5))

        # Distribution plot
        plt.subplot(1, 2, 1)
        plt.hist(val_predictions.numpy(), bins=30, alpha=0.7)
        plt.title(f'D Coefficient Distribution - Warm-up Epoch {epoch}')

        # Learning rate plot
        plt.subplot(1, 2, 2)
        plt.plot(lr_values)
        plt.title('Learning Rate Cycle')
        plt.xlabel('Batch')
        plt.ylabel('Learning Rate')

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'warmup_epoch_{epoch}.png'))
        plt.close()

    # Save warm-up model
    torch.save({
        'phase': 'warmup',
        'model_state_dict': model.state_dict(),
    }, os.path.join(log_dir, 'warmup_checkpoint.pth'))

    print(f"\nWarm-up completed. Proceeding with curriculum training.")

    # Training phases with genetic exploration and cyclic learning rates
    # First, train each coefficient individually with increasing bounds
    for bound in np.arange(0, 5, 0.5):
        print(f"\n=== Starting training phase with bound {bound:.1f} ===")

        for coeff_idx in range(8):
            folder_name = f'Training{param_names[coeff_idx]}{bound:.1f}'.replace('.', '_')
            print(f"\nTraining on {folder_name}")

            # Get data loaders for current phase
            train_loader, val_loader = create_curriculum_loader(folder_name, [coeff_idx])
            batches_per_epoch = len(train_loader)

            # Initialize optimizer with base learning rate for cycling
            base_lr, max_lr = 1e-5, 5e-4  # Slightly lower for main training
            optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

            best_val_loss = float('inf')
            patience_counter = 0

            # Training loop for current phase
            for epoch in range(max_epochs_per_phase):
                # Update epoch in criterion
                criterion.update_epoch(epoch)

                # Check if we should run genetic exploration this epoch
                if epoch > 0 and epoch % genetic_exploration_frequency == 0:
                    print(f"\n=== Running genetic exploration at epoch {epoch} ===")
                    model = train_with_genetic_exploration(
                        model,
                        train_loader,
                        lambda outputs, targets: criterion(outputs[:, coeff_idx].unsqueeze(1),
                                                           targets[:, coeff_idx].unsqueeze(1)),
                        population_size=population_size,
                        batch_samples=10
                    )
                    # Update optimizer to use the new model parameters
                    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

                model.train()
                train_losses = []
                lr_values = []

                # Training phase
                for batch_idx, (images, targets) in enumerate(train_loader):
                    # Update learning rate for this batch
                    current_lr = get_cyclic_lr(epoch, batch_idx, batches_per_epoch, base_lr, max_lr)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr

                    lr_values.append(current_lr)

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
                              f"Loss: {loss.item():.6f} LR: {current_lr:.6f}", end='')

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
                avg_lr = np.mean(lr_values)

                # Analyze predictions
                val_predictions = torch.cat(val_predictions)
                val_targets = torch.cat(val_targets_list)

                print(f"\nEpoch {epoch} Summary:")
                print(f"Training Loss: {train_loss:.6f}")
                print(f"Validation Loss: {val_loss:.6f}")
                print(f"Average LR: {avg_lr:.6f}")
                print(f"Predictions range: [{val_predictions.min():.3f}, {val_predictions.max():.3f}]")
                print(f"Targets range: [{val_targets.min():.3f}, {val_targets.max():.3f}]")
                print(f"Patience: {patience_counter}")

                # Plot prediction distribution and learning rate
                plt.figure(figsize=(12, 5))

                # Distribution plot
                plt.subplot(1, 2, 1)
                plt.hist(val_predictions.numpy(), bins=50, alpha=0.5, label='Predictions')
                plt.hist(val_targets.numpy(), bins=50, alpha=0.5, label='Targets')
                plt.title(f'Distribution for {param_names[coeff_idx]} - Epoch {epoch}')
                plt.legend()

                # Learning rate plot
                plt.subplot(1, 2, 2)
                plt.plot(lr_values)
                plt.title('Learning Rate Cycle')
                plt.xlabel('Batch')
                plt.ylabel('Learning Rate')

                plt.tight_layout()
                plt.savefig(os.path.join(log_dir, f'{param_names[coeff_idx]}_epoch_{epoch}.png'))
                plt.close()

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
    # Also using cyclic learning rates and genetic exploration
    print("\n=== Starting final phase with TrainingEverything ===")

    for num_active_coeffs in range(1, 9):
        active_coeffs = list(range(num_active_coeffs))
        print(f"\nTraining on coefficients: {[param_names[i] for i in active_coeffs]}")

        train_loader, val_loader = create_curriculum_loader('TrainingEverything', active_coeffs)
        batches_per_epoch = len(train_loader)

        base_lr, max_lr = 1e-5, 3e-4  # Reduced max LR for multi-coefficient training
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs_per_phase):
            criterion.update_epoch(epoch)

            # Check if we should run genetic exploration this epoch
            if epoch > 0 and epoch % genetic_exploration_frequency == 0:
                print(f"\n=== Running genetic exploration at epoch {epoch} ===")
                model = train_with_genetic_exploration(
                    model,
                    train_loader,
                    lambda outputs, targets: criterion(outputs[:, active_coeffs], targets[:, active_coeffs]),
                    population_size=population_size,
                    batch_samples=10
                )
                # Update optimizer after genetic exploration
                optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

            model.train()
            train_losses = []
            lr_values = []

            # Training phase
            for batch_idx, (images, targets) in enumerate(train_loader):
                # Update learning rate for this batch
                current_lr = get_cyclic_lr(epoch, batch_idx, batches_per_epoch, base_lr, max_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                lr_values.append(current_lr)

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
                          f"Loss: {loss.item():.6f} LR: {current_lr:.6f}", end='')

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
            avg_lr = np.mean(lr_values)

            # Analyze predictions
            val_predictions = torch.cat(val_predictions)
            val_targets = torch.cat(val_targets_list)

            print(f"\nEpoch {epoch} Summary:")
            print(f"Training Loss: {train_loss:.6f}")
            print(f"Validation Loss: {val_loss:.6f}")
            print(f"Average LR: {avg_lr:.6f}")

            # Plot prediction distributions and learning rate
            plt.figure(figsize=(15, 5 * ((num_active_coeffs + 1) // 2) + 5))

            # Coefficient distributions
            for i, coeff_idx in enumerate(active_coeffs):
                plt.subplot(((num_active_coeffs + 1) // 2) + 1, 2, i + 1)
                plt.hist(val_predictions[:, i].numpy(), bins=50, alpha=0.5, label='Predictions')
                plt.hist(val_targets[:, i].numpy(), bins=50, alpha=0.5, label='Targets')
                plt.title(f'Distribution for {param_names[coeff_idx]}')
                plt.legend()

            # Learning rate plot (in the last position)
            plt.subplot(((num_active_coeffs + 1) // 2) + 1, 2, num_active_coeffs + 1)
            plt.plot(lr_values)
            plt.title('Learning Rate Cycle')
            plt.xlabel('Batch')
            plt.ylabel('Learning Rate')

            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f'final_phase_epoch_{epoch}.png'))
            plt.close()

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