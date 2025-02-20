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
import cv2
from scipy.fftpack import fft2, fftshift
from scipy.special import factorial
from sklearn.preprocessing import StandardScaler

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unwrap_phase_gpu(phase):
    """GPU-based phase unwrapping."""
    # Simple unwrapping - you might want a more sophisticated version
    pi = torch.tensor(math.pi, device=phase.device)
    unwrapped = torch.where(phase < 0, phase + 2 * pi, phase)
    return unwrapped

class GPUFeatureExtractor(nn.Module):
    def __init__(self, n_terms=10):
        super().__init__()
        self.n_terms = n_terms

        # Pre-compute factorials on CPU once, then move to GPU
        max_n = n_terms + 1
        factorials = torch.tensor([math.factorial(i) for i in range(max_n)],
                                  dtype=torch.float32).cuda()
        self.register_buffer('factorials', factorials)

    def factorial(self, n):
        """GPU-based factorial lookup."""
        return self.factorials[n]

    def zernike_polynomial_gpu(self, n, m, rho, theta):
        """
        Compute Zernike polynomial on GPU.
        All inputs should be GPU tensors.
        """
        if (n - abs(m)) % 2 != 0:
            return torch.zeros_like(rho, device=rho.device)

        R = torch.zeros_like(rho, device=rho.device)
        for k in range((n - abs(m)) // 2 + 1):
            # Calculate coefficient using pre-computed factorials
            c = ((-1) ** k) * self.factorial(n - k) / (
                    self.factorial(k) *
                    self.factorial((n + abs(m)) // 2 - k) *
                    self.factorial((n - abs(m)) // 2 - k)
            )
            R += c * rho ** (n - 2 * k)

        if m >= 0:
            return R * torch.cos(m * theta)
        else:
            return R * torch.sin(-m * theta)

    def compute_zernike_coefficients_gpu(self, phase_map):
        """Compute Zernike coefficients entirely on GPU."""
        batch_size = phase_map.shape[0]
        height, width = phase_map.shape[2], phase_map.shape[3]

        # Create coordinate grid on GPU
        y = torch.linspace(-1, 1, height, device=phase_map.device)
        x = torch.linspace(-1, 1, width, device=phase_map.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

        # Compute polar coordinates on GPU
        rho = torch.sqrt(grid_x**2 + grid_y**2)
        theta = torch.atan2(grid_y, grid_x)

        # Create mask for valid region
        mask = (rho <= 1.0).float()

        # Expand dimensions for batch processing
        rho = rho.expand(batch_size, 1, height, width)
        theta = theta.expand(batch_size, 1, height, width)
        mask = mask.expand(batch_size, 1, height, width)

        coefficients = []
        for n in range(self.n_terms):
            for m in range(-n, n + 1, 2):
                Z = self.zernike_polynomial_gpu(n, m, rho, theta)
                # Compute coefficient using masked operations
                coeff = torch.sum(phase_map * Z * mask, dim=(2,3)) / torch.sum(mask, dim=(2,3))
                coefficients.append(coeff)

        return torch.cat(coefficients, dim=1)

    def extract_phase_gradient_gpu(self, phase):
        """Extract phase gradients using GPU operations."""
        # Use torch.gradient for GPU-accelerated gradient computation
        grad_y, grad_x = torch.gradient(phase, dim=(2,3))
        return grad_x, grad_y

    def forward(self, x):
        """
        Process a batch of interferograms entirely on GPU.
        x: tensor of shape [batch_size, 1, height, width]
        """
        batch_size = x.shape[0]

        # FFT on GPU
        fft_image = torch.fft.rfft2(x)
        magnitude = torch.abs(fft_image)
        phase = torch.angle(fft_image)

        # Unwrap phase on GPU
        # Note: This is a simple unwrap, you might want to implement a more sophisticated version
        phase_unwrapped = torch.where(
            phase < 0,
            phase + 2 * math.pi,
            phase
        )

        # Extract features
        grad_x, grad_y = self.extract_phase_gradient_gpu(phase_unwrapped)
        zernike_coeffs = self.compute_zernike_coefficients_gpu(phase_unwrapped)

        # Compute statistics on GPU
        slope_stats = torch.cat([
            grad_x.mean(dim=(2,3)),
            grad_y.mean(dim=(2,3)),
            grad_x.std(dim=(2,3)),
            grad_y.std(dim=(2,3))
        ], dim=1)

        # Compute fringe frequency using FFT magnitude
        freq_features = torch.max(torch.max(magnitude, dim=2)[0], dim=2)[0]

        # Combine all features
        features = torch.cat([
            zernike_coeffs,        # [batch_size, n_zernike_coeffs]
            slope_stats,           # [batch_size, 4]
            freq_features          # [batch_size, 1]
        ], dim=1)

        return features

def replace_feature_extractor(model):
    """Replace the existing feature extractor with the GPU version."""
    model.feature_extractor = GPUFeatureExtractor(n_terms=10).cuda()
    return model

class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        batch, c, h, w = x.size()
        y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).repeat(batch, 1, 1, w)
        x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).repeat(batch, 1, h, 1)
        r = torch.sqrt(x_coords**2 + y_coords**2)
        theta = torch.atan2(y_coords, x_coords)

        pos_enc = []
        for freq in [1, 2, 4, 8]:
            pos_enc.append(torch.sin(r * math.pi * freq))
            pos_enc.append(torch.cos(r * math.pi * freq))

        for freq in [1, 2, 4]:
            pos_enc.append(torch.sin(theta * freq))
            pos_enc.append(torch.cos(theta * freq))

        pos_enc.append(torch.sin(x_coords * math.pi))
        pos_enc.append(torch.cos(x_coords * math.pi))
        pos_enc.append(torch.sin(y_coords * math.pi))
        pos_enc.append(torch.cos(y_coords * math.pi))

        pos_encoding = torch.cat(pos_enc, dim=1)
        return torch.cat([x, pos_encoding], dim=1)

class FourierConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_fft = torch.fft.rfft2(x)
        real, imag = x_fft.real, x_fft.imag

        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)

        x_fft_out = torch.complex(real_out, imag_out)
        output_size = (x.shape[2] // self.stride, x.shape[3] // self.stride)
        x_out = torch.fft.irfft2(x_fft_out, s=output_size)

        return self.bn(x_out)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_fourier=False):
        super(ResidualBlock, self).__init__()

        # Main convolution path
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

def create_curriculum_loader(folder_path, active_coeffs, batch_size=100, num_workers=4):
    """
    Create a dataloader for a specific curriculum phase.

    Args:
        folder_path (str): Path to the training folder
        active_coeffs (list): List of indices of coefficients to train on
        batch_size (int): Batch size for the dataloader
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create dataset
    dataset = CombinedCurriculumDataset(folder_path, active_coeffs)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader

class FeatureExtractionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        # Initialize GPU feature extractor
        self.gpu_extractor = GPUFeatureExtractor(n_terms=10)
        self.n_features = 60  # Same as before

    def extract_phase_gradient(self, interferogram):
        """Compute phase gradients using Fourier transform method."""
        if len(interferogram.shape) == 3:
            interferogram = cv2.cvtColor(interferogram, cv2.COLOR_BGR2GRAY)

        interferogram = interferogram.astype(np.float32) / 255.0
        fft_image = fftshift(fft2(interferogram))
        phase = np.angle(fft_image)

        grad_x = np.gradient(phase, axis=1)
        grad_y = np.gradient(phase, axis=0)
        return grad_x, grad_y

    def compute_fringe_frequency(self, interferogram):
        """Compute dominant fringe frequency using Fourier analysis."""
        if len(interferogram.shape) == 3:
            interferogram = cv2.cvtColor(interferogram, cv2.COLOR_BGR2GRAY)

        interferogram = interferogram.astype(np.float32) / 255.0
        window = np.outer(np.hanning(interferogram.shape[0]),
                          np.hanning(interferogram.shape[1]))
        interferogram_windowed = interferogram * window

        fft_image = np.abs(fftshift(fft2(interferogram_windowed)))
        mask = np.ones_like(fft_image)
        center_region = 5
        center_y, center_x = fft_image.shape[0]//2, fft_image.shape[1]//2
        mask[center_y-center_region:center_y+center_region,
        center_x-center_region:center_x+center_region] = 0

        peak_coords = np.unravel_index(np.argmax(fft_image * mask),
                                       fft_image.shape)
        freq_y = peak_coords[0] - fft_image.shape[0] // 2
        freq_x = peak_coords[1] - fft_image.shape[1] // 2

        return np.sqrt(freq_x**2 + freq_y**2)

    def forward(self, x):
        """
        Process a batch of interferograms using GPU operations.
        x: tensor of shape [batch_size, 1, height, width]
        """
        # Use the GPU extractor directly
        features = self.gpu_extractor(x)

        # Scale features (keep this as is since it's lightweight)
        if not hasattr(self, 'fitted'):
            self.scaler.fit(features.cpu().numpy())
            self.fitted = True
        features_scaled = torch.tensor(
            self.scaler.transform(features.cpu().numpy()),
            device=x.device
        )

        return features_scaled

class CombinedInterferogramNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extraction path
        self.feature_extractor = FeatureExtractionModule()

        # CNN path with initial convolution and positional encoding
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.positional_encoding = SinusoidalPositionalEncoding(32)
        self.post_positional = nn.Sequential(
            nn.Conv2d(50, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Residual blocks with Fourier processing
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, use_fourier=False),
            ResidualBlock(64, 64, use_fourier=True)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_fourier=True),
            ResidualBlock(128, 128, use_fourier=True)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_fourier=True),
            ResidualBlock(256, 256, use_fourier=True)
        )

        # Global pooling
        self.global_pool_avg = nn.AdaptiveAvgPool2d(1)
        self.global_pool_max = nn.AdaptiveMaxPool2d(1)

        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Linear(512 + self.feature_extractor.n_features, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )

        # Separate predictors with shared fusion features
        self.defocus_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

        self.tilt_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2)
        )

        self.astigmatism_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2)
        )

        self.higher_order_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 3)
        )

        self.apply(self._init_weights)
        self.forward_count = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.4)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -0.1, 0.1)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='leaky_relu', a=0.2)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.Linear) and hasattr(m, 'out_features'):
            if m.out_features <= 3:
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.2, 0.2)

    def forward(self, x):
        # Extract classical features
        classical_features = self.feature_extractor(x)

        # CNN path
        x = self.initial(x)
        x = self.positional_encoding(x)
        x = self.post_positional(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global pooling
        x_avg = self.global_pool_avg(x).view(x.size(0), -1)
        x_max = self.global_pool_max(x).view(x.size(0), -1)
        x_pooled = torch.cat([x_avg, x_max], dim=1)

        # Combine CNN features with classical features
        combined_features = torch.cat([x_pooled, classical_features], dim=1)

        # Feature fusion
        fused_features = self.fusion(combined_features)

        # Separate predictions through specialized pathways
        defocus_out = self.defocus_predictor(fused_features)            # D (index 0)
        tilt_out = self.tilt_predictor(fused_features)                  # B, C (indices 2, 1)
        astigmatism_out = self.astigmatism_predictor(fused_features)    # E, I (indices 6, 7)
        higher_out = self.higher_order_predictor(fused_features)        # G, F, J (indices 3, 4, 5)

        # Reorder outputs to match expected [D, C, B, G, F, J, E, I] format
        outputs = torch.cat([
            defocus_out,                  # D (0)
            tilt_out[:, 1:2],            # C (1)
            tilt_out[:, 0:1],            # B (2)
            higher_out[:, 0:1],          # G (3)
            higher_out[:, 1:2],          # F (4)
            higher_out[:, 2:3],          # J (5)
            astigmatism_out[:, 0:1],     # E (6)
            astigmatism_out[:, 1:2],     # I (7)
        ], dim=1)

        if self.forward_count < 5:
            print(f"\nForward pass {self.forward_count}")
            print(f"Classical features shape: {classical_features.shape}")
            print(f"CNN features shape: {x_pooled.shape}")
            print(f"Combined features shape: {combined_features.shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            if torch.cuda.is_available():
                print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
            self.forward_count += 1

        return outputs

class CombinedCurriculumDataset(Dataset):
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

        # Load all image paths
        self.file_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .jpg files found in {folder_path}")

        print(f"Found {len(self.file_paths)} images")

        # Calculate statistics on a subset of images
        sample_size = min(1000, len(self.file_paths))
        sample_indices = np.random.choice(len(self.file_paths), sample_size, replace=False)
        sample_params = []

        for idx in sample_indices:
            img_path = self.file_paths[idx]
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

        # Load and transform image
        image = Image.open(img_path)
        image = self.transform(image)

        # Extract parameters
        params = self._extract_params_from_filename(img_path)

        # Apply data augmentation with noise
        if torch.rand(1).item() < 0.3:  # 30% chance of adding noise
            noise = torch.randn_like(image) * 0.03
            image = torch.clamp(image + noise, -1, 1)

        return image, params

class CombinedPhaseAwareLoss(nn.Module):
    def __init__(self, similarity_threshold=0.1, diversity_weight=0.1,
                 temperature=1.0, spread_factor=0.3, warm_up_epochs=5):
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

    def flat_expected_loss(self, pred, target):
        # Absolute error component
        abs_error = torch.abs(pred - target)

        # Correction term to ensure constant expected loss
        correction_term = 0.5 * (1 - pred**2)

        # Penalize predictions outside [-1, 1] range
        outside_range = torch.clamp(torch.abs(pred) - 1.0, min=0)
        out_of_bounds_penalty = outside_range**2 * 10 + .2

        # Combine all components
        total_loss = abs_error + correction_term + out_of_bounds_penalty
        return total_loss

    def forward(self, pred, target, is_training=False):
        # Ensure inputs are at least 2D
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        num_coeffs = pred.shape[1]
        batch_size = pred.shape[0]

        # Use appropriate weights
        if num_coeffs == 1:
            coeff_weights = torch.ones(1, device=pred.device)
        else:
            coeff_weights = self.coeff_weights[:num_coeffs].to(pred.device)

        # Main loss with coefficient weighting
        base_loss = self.flat_expected_loss(pred, target)
        weighted_base_loss = (base_loss * coeff_weights).mean()

        # Identify predictions within threshold of target
        close_to_target = torch.abs(pred - target) < 0.2

        # Calculate standard deviation
        pred_std = pred.std(dim=0).mean()
        spread_loss = torch.exp(-pred_std / self.spread_factor)

        # Initialize diversity loss
        diversity_penalty = torch.zeros(batch_size, device=pred.device)
        current_weight = 0.0

        if not close_to_target.all():
            if num_coeffs > 1:
                not_close_mask = ~close_to_target.all(dim=1)
                if not_close_mask.any():
                    pred_for_diversity = pred[not_close_mask]
                    if len(pred_for_diversity) > 1:
                        scaled_pred = pred_for_diversity / self.temperature
                        diff_matrix = torch.cdist(scaled_pred, scaled_pred, p=1)
                        too_similar = diff_matrix < (self.similarity_threshold * num_coeffs)
                        too_similar.fill_diagonal_(False)
                        similarity_counts = too_similar.float().sum(dim=1)
                        not_close_penalty = torch.exp(similarity_counts / batch_size) - 1
                        diversity_penalty[not_close_mask] = not_close_penalty
                        current_weight = self.diversity_weight * (1 - torch.exp(-similarity_counts.mean()))
            else:
                not_close_mask = ~close_to_target.squeeze()
                if not_close_mask.any():
                    pred_for_diversity = pred[not_close_mask].squeeze()
                    if len(pred_for_diversity) > 1:
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

        # Apply epoch-dependent weighting
        epoch_factor = min(1.0, self.epoch / 20)
        spread_weight = self.spread_factor * epoch_factor

        # Combine all loss components
        total_loss = (weighted_base_loss +
                      (current_weight * diversity_loss) +
                      (spread_weight * spread_loss))

        # Debug prints
        if torch.rand(1).item() < 0.01:
            print(f"\nLoss Components (Epoch {self.epoch}):")
            print(f"Weighted Base Loss: {weighted_base_loss:.6f}")
            print(f"Diversity Loss: {diversity_loss:.6f} (weight: {current_weight:.3f})")
            print(f"Spread Loss: {spread_loss:.6f} (weight: {spread_weight:.3f})")
            print(f"Prediction std: {pred_std:.4f}")
            print(f"Total Loss: {total_loss:.6f}")
            print(f"Predictions close to target: {close_to_target.sum().item()}/{batch_size*num_coeffs}")
            print(f"Predictions mean: {pred.mean():.3f}, std: {pred.std():.3f}")

            if num_coeffs > 1:
                for i in range(num_coeffs):
                    coeff_vals = pred[:, i]
                    print(f"  Coeff {i}: mean={coeff_vals.mean():.3f}, std={coeff_vals.std():.3f}")
                    print(f"     range: [{coeff_vals.min():.3f}, {coeff_vals.max():.3f}]")

        if self.epoch < self.warm_up_epochs and is_training:
            min_prediction_magnitude = 0.8 * (1 - self.epoch / self.warm_up_epochs)

            # Lower bound penalty
            outside_zero_penalty = torch.clamp(min_prediction_magnitude - torch.abs(pred), min=0)

            # Upper bound penalty
            outside_range_penalty = torch.clamp(torch.abs(pred) - 1.0, min=0)

            # Range coverage penalty
            max_vals = torch.max(pred, dim=0)[0]
            min_vals = torch.min(pred, dim=0)[0]
            range_coverage_penalty = (torch.clamp(0.5 - max_vals, min=0) +
                                      torch.clamp(min_vals + 0.5, min=0))

            # Combine all penalties
            zero_constraint_loss = (10 * (outside_zero_penalty + outside_range_penalty) +
                                    5 * range_coverage_penalty)
            total_loss = zero_constraint_loss.mean() * 100

        return total_loss

    def update_epoch(self, epoch):
        self.epoch = epoch

def train_combined_curriculum():
    print(f"Starting combined curriculum training on {device}")
    start_time = time.time()

    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'combined_curriculum_training_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)

    # Training settings
    target_loss = 0.015
    patience_limit = 100
    max_epochs_per_phase = 1000
    param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']

    # Initialize model and criterion
    model = CombinedInterferogramNet().to(device)
    criterion = CombinedPhaseAwareLoss(
        similarity_threshold=0.25,
        diversity_weight=1.0,
        temperature=0.8,
        spread_factor=0.3
    )

    # Warm-up phase
    print("\n=== Starting warm-up phase ===")
    warm_up_epochs = 1
    train_loader, val_loader = create_curriculum_loader('TrainingD0_0', [0])

    base_lr, max_lr = 5e-5, 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    train_with_curriculum_phase(
        model, optimizer, criterion, train_loader, val_loader,
        warm_up_epochs, log_dir, 'warmup', target_loss, patience_limit,
        base_lr, max_lr
    )

    # Save warm-up model
    # Save warm-up model
    torch.save({
        'phase': 'warmup',
        'model_state_dict': model.state_dict(),
    }, os.path.join(log_dir, 'warmup_checkpoint.pth'))

    # Training phases with progressive complexity
    for bound in np.arange(0, 5, 0.5):
        print(f"\n=== Starting training phase with bound {bound:.1f} ===")

        for coeff_idx in range(8):
            folder_name = f'Training{param_names[coeff_idx]}{bound:.1f}'.replace('.', '_')
            print(f"\nTraining on {folder_name}")

            train_loader, val_loader = create_curriculum_loader(folder_name, [coeff_idx])
            base_lr, max_lr = 1e-5, 5e-4
            optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

            train_with_curriculum_phase(
                model, optimizer, criterion, train_loader, val_loader,
                max_epochs_per_phase, log_dir, f'bound{bound:.1f}_{param_names[coeff_idx]}',
                target_loss, patience_limit, base_lr, max_lr
            )

    # Final phase with all coefficients
    print("\n=== Starting final phase with all coefficients ===")

    for num_active_coeffs in range(1, 9):
        active_coeffs = list(range(num_active_coeffs))
        print(f"\nTraining on coefficients: {[param_names[i] for i in active_coeffs]}")

        train_loader, val_loader = create_curriculum_loader('TrainingEverything', active_coeffs)
        base_lr, max_lr = 1e-5, 3e-4
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

        train_with_curriculum_phase(
            model, optimizer, criterion, train_loader, val_loader,
            max_epochs_per_phase, log_dir, f'final_phase_{num_active_coeffs}coeffs',
            target_loss, patience_limit, base_lr, max_lr
        )

    print("\nCombined curriculum training completed!")
    print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")

    return model

def train_with_curriculum_phase(model, optimizer, criterion, train_loader, val_loader,
                                max_epochs, log_dir, phase_name, target_loss, patience_limit,
                                base_lr, max_lr):
    """Train a single phase of the curriculum."""

    best_val_loss = float('inf')
    patience_counter = 0
    batches_per_epoch = len(train_loader)

    for epoch in range(max_epochs):
        criterion.update_epoch(epoch)
        model.train()
        train_losses = []
        lr_values = []

        # Training phase
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Update learning rate
            current_lr = get_cyclic_lr(epoch, batch_idx, batches_per_epoch, base_lr, max_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            lr_values.append(current_lr)

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, targets, is_training=True)
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
                val_loss = criterion(outputs, targets)

                val_losses.append(val_loss.item())
                val_predictions.append(outputs.cpu())
                val_targets_list.append(targets.cpu())

        # Compute statistics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_predictions = torch.cat(val_predictions)
        val_targets = torch.cat(val_targets_list)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"Training Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Predictions range: [{val_predictions.min():.3f}, {val_predictions.max():.3f}]")
        print(f"Targets range: [{val_targets.min():.3f}, {val_targets.max():.3f}]")

        # Plot distributions
        plt.figure(figsize=(12, 6))
        for i in range(val_predictions.shape[1]):
            plt.subplot(2, 4, i+1)
            plt.hist(val_predictions[:, i].numpy(), bins=30, alpha=0.5, label='Pred')
            plt.hist(val_targets[:, i].numpy(), bins=30, alpha=0.5, label='Target')
            plt.title(f'Coeff {i}')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{phase_name}_epoch{epoch}.png'))
        plt.close()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save checkpoint
            torch.save({
                'phase': phase_name,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(log_dir, f'checkpoint_{phase_name}.pth'))

        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\nEarly stopping for phase {phase_name}")
                break

        # Check if we've reached target loss
        if val_loss < target_loss:
            print(f"\nReached target loss for phase {phase_name}!")
            break

def get_cyclic_lr(epoch, batch_idx, batches_per_epoch, base_lr=1e-4, max_lr=1e-3):
    """Compute cyclic learning rate."""
    cycle_length = 10  # epochs
    cycle = (epoch + batch_idx/batches_per_epoch) / cycle_length
    cycle_position = cycle - int(cycle)

    if cycle_position < 0.5:
        # Increasing phase
        return base_lr + (max_lr - base_lr) * (2 * cycle_position)
    else:
        # Decreasing phase
        return max_lr - (max_lr - base_lr) * (2 * (cycle_position - 0.5))

def predict_coefficients(model, image_path):
    """
    Predict aberration coefficients for a single interferogram image.

    Args:
        model: Trained CombinedInterferogramNet model
        image_path: Path to interferogram image

    Returns:
        Dictionary containing predicted coefficients
    """
    # Transform for single image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(1),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load and transform image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move to device and predict
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        predictions = model(image)

        # Scale predictions back to original range ([-5, 5])
        predictions = predictions * 5.0

    # Format predictions
    param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']
    results = {}
    for name, value in zip(param_names, predictions[0].cpu().numpy()):
        results[name] = float(value)

    return results

def analyze_model_performance(model, test_loader, criterion):
    """
    Analyze model performance on test set.

    Args:
        model: Trained CombinedInterferogramNet model
        test_loader: DataLoader for test set
        criterion: Loss criterion

    Returns:
        Dictionary containing performance metrics
    """
    model.eval()
    test_losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            test_losses.append(loss.item())
            predictions.append(output.cpu())
            targets.append(target.cpu())

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    # Calculate metrics
    param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']
    metrics = {
        'overall_loss': np.mean(test_losses),
        'coefficient_metrics': {}
    }

    for i, name in enumerate(param_names):
        pred = predictions[:, i]
        targ = targets[:, i]

        metrics['coefficient_metrics'][name] = {
            'mean_error': float((pred - targ).abs().mean()),
            'std_error': float((pred - targ).std()),
            'prediction_range': [float(pred.min()), float(pred.max())],
            'target_range': [float(targ.min()), float(targ.max())]
        }

    return metrics

def main():
    """Main execution function."""
    print("Starting Combined Interferogram Analysis System")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Training parameters
    params = {
        'batch_size': 100,
        'num_workers': 64,
        'pin_memory': True
    }

    # Initialize model and criterion
    model = CombinedInterferogramNet().to(device)
    model = replace_feature_extractor(model)
    criterion = CombinedPhaseAwareLoss(
        similarity_threshold=0.25,
        diversity_weight=1.0,
        temperature=0.8,
        spread_factor=0.3
    )

    # Train model using curriculum learning
    model = train_combined_curriculum()

    # Create test dataset and loader
    test_dataset = CombinedCurriculumDataset('TestingEverything', list(range(8)))
    test_loader = DataLoader(test_dataset, **params)

    # Analyze model performance
    print("\nAnalyzing model performance...")
    performance_metrics = analyze_model_performance(model, test_loader, criterion)

    # Print performance summary
    print("\nPerformance Summary:")
    print(f"Overall Loss: {performance_metrics['overall_loss']:.6f}")
    print("\nCoefficient-wise Metrics:")
    for coeff, metrics in performance_metrics['coefficient_metrics'].items():
        print(f"\nCoefficient {coeff}:")
        print(f"  Mean Error: {metrics['mean_error']:.6f}")
        print(f"  Std Error: {metrics['std_error']:.6f}")
        print(f"  Prediction Range: [{metrics['prediction_range'][0]:.3f}, {metrics['prediction_range'][1]:.3f}]")
        print(f"  Target Range: [{metrics['target_range'][0]:.3f}, {metrics['target_range'][1]:.3f}]")

    # Save final model and metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'combined_model_final_{timestamp}'
    os.makedirs(save_path, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'performance_metrics': performance_metrics,
        'training_timestamp': timestamp
    }, os.path.join(save_path, 'final_model.pth'))

    print(f"\nModel and metrics saved to {save_path}")
    print("\nTraining and evaluation completed successfully!")

if __name__ == '__main__':
    main()
    #