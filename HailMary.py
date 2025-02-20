import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime

class PretrainedInterferogramNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Replace first conv layer to accept single channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        return self.resnet(x)

class InterferogramDataset(Dataset):
    def __init__(self, folder_path, active_coeffs):
        self.folder_path = folder_path
        self.active_coeffs = active_coeffs
        self.file_paths = glob.glob(os.path.join(folder_path, '*.jpg'))

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # ImageNet stats for single channel
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]

        # Load and transform image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = self.transform(image)

        # Extract parameters from filename
        params = self._extract_params_from_filename(img_path)

        return image, params[self.active_coeffs], img_path

    def _extract_params_from_filename(self, img_path):
        name = os.path.splitext(os.path.basename(img_path))[0]
        name = name.replace('n', '-').replace('p', '.')
        parts = name.split('_')[1:9]
        params = np.zeros(8)
        for i, part in enumerate(parts):
            params[i] = float(part[1:])
        return torch.FloatTensor(params)

def create_prediction_plot(predictions, targets, param_names, save_dir, phase, epoch):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create subplot for each coefficient
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    for i, (name, ax) in enumerate(zip(param_names, axes)):
        if i < len(predictions[0]):
            ax.scatter(targets[:, i].cpu(), predictions[:, i].cpu(), alpha=0.5)
            ax.plot([-0.5, 0.5], [-0.5, 0.5], 'r--')
            ax.set_title(f'{name} Coefficient')
            ax.set_xlabel('True Value')
            ax.set_ylabel('Predicted Value')
            ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'predictions_{phase}_epoch_{epoch}_{timestamp}.png'))
    plt.close()

def validate_with_images(model, val_loader, criterion, device, param_names, save_dir, epoch):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    examples_dir = os.path.join(save_dir, 'example_predictions')
    os.makedirs(examples_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, targets, img_paths) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())

            # Save example predictions every 50 batches
            if batch_idx % 50 == 51:
                for i in range(min(5, len(images))):
                    img = images[i].cpu().numpy()
                    pred = outputs[i].cpu().numpy()
                    target = targets[i].cpu().numpy()

                    plt.figure(figsize=(10, 5))

                    # Plot interferogram
                    plt.subplot(1, 2, 1)
                    plt.imshow(img[0], cmap='gray')
                    plt.title('Input Interferogram')

                    # Plot coefficients
                    plt.subplot(1, 2, 2)
                    num_coeffs = len(pred)  # Get number of coefficients we're actually using
                    x = np.arange(num_coeffs)  # Create x array of correct length

                    plt.bar(x - 0.2, target, 0.4, label='Target', alpha=0.8)
                    plt.bar(x + 0.2, pred, 0.4, label='Prediction', alpha=0.8)
                    plt.xticks(x, param_names[:num_coeffs])  # Use only the names we need
                    plt.legend()
                    plt.title(f'Coefficient Values ({num_coeffs} active)')

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    plt.savefig(os.path.join(examples_dir,
                                             f'example_epoch_{epoch}_batch_{batch_idx}_img_{i}_{timestamp}.png'))
                    plt.close()

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    # Get number of coefficients for correlation plots
    num_coeffs = predictions.shape[1]
    create_prediction_plot(predictions, targets, param_names[:num_coeffs], save_dir, 'validation', epoch)

    coeff_errors = torch.abs(predictions - targets).mean(dim=0)
    return total_loss / len(val_loader), coeff_errors

def create_curriculum_loader(folder_path, active_coeffs, batch_size=32):
    """Create data loaders for curriculum training"""
    dataset = InterferogramDataset(folder_path, active_coeffs)

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
        num_workers=8,  # Reduced from 64 to prevent memory issues
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Changed to False for validation
        num_workers=8,
        pin_memory=True
    )

    return train_loader, val_loader

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    for batch_idx, (images, targets, _) in enumerate(train_loader):  # Added _ for img_paths
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}')

    return total_loss / num_batches
    param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']
    criterion = nn.MSELoss()  # Using simple MSE loss for stability

    # Training loop similar to before but with image saving
    for param_idx, param_name in enumerate(param_names):
        print(f"\n=== Training {param_name} coefficient ===")
        for bound in np.arange(0, 0.5, 0.05):
            folder_name = f'Training{param_name}{bound:.1f}'.replace('.', '_')
            train_loader, val_loader = create_curriculum_loader(folder_name, [param_idx], batch_size=32)

            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

            for epoch in range(100):  # Reduced epochs per phase
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, errors = validate_with_images(model, val_loader, criterion, device,
                                                        [param_names[param_idx]], save_dir, epoch)

                scheduler.step(val_loss)

                # Save model
                if val_loss < 0.015:
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, f'model_{param_name}_{bound:.2f}.pth'))
                    break

    # Final phase training
    print("\n=== Training on full coefficient set ===")
    train_loader, val_loader = create_curriculum_loader('TrainingEverything',
                                                        list(range(8)), batch_size=32)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')  # Initialize best_val_loss

    for epoch in range(200):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, errors = validate_with_images(model, val_loader, criterion, device,
                                                [param_names[param_idx]], save_dir, epoch)

        scheduler.step(val_loss)

        # Save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

def train_curriculum(model, base_folder, device, save_dir='training_outputs'):
    """
    Implements curriculum learning strategy for interferogram analysis
    """
    # Initialize parameters and paths
    param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']
    criterion = nn.MSELoss()
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nStarting curriculum training, saving outputs to {save_dir}")

    try:
        # Phase 1: Individual coefficient training
        for param_idx, param_name in enumerate(param_names):
            print(f"\n=== Training {param_name} coefficient ===")
            param_save_dir = os.path.join(save_dir, f'param_{param_name}')
            os.makedirs(param_save_dir, exist_ok=True)

            # Train with increasing bounds
            for bound in np.arange(0, 5, 0.5):
                folder_name = f'Training{param_name}{bound:.1f}'.replace('.', '_')
                print(f"\nTraining on {folder_name}")

                # Create data loaders
                try:
                    train_loader, val_loader = create_curriculum_loader(
                        folder_name,
                        [param_idx],
                        batch_size=32
                    )
                except (FileNotFoundError, RuntimeError) as e:
                    print(f"Error loading data from {folder_name}: {e}")
                    continue

                # Initialize optimizer with larger learning rate for bigger bounds
                optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

                best_val_loss = float('inf')
                epochs_without_improvement = 0

                # Training loop for current coefficient and bound
                for epoch in range(100):
                    print(f"\nEpoch {epoch}")

                    # Train
                    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

                    # Validate and save visualizations
                    val_loss, errors = validate_with_images(
                        model,
                        val_loader,
                        criterion,
                        device,
                        [param_names[param_idx]],
                        param_save_dir,
                        epoch
                    )

                    # Print metrics
                    print(f"Training loss: {train_loss:.6f}")
                    print(f"Validation loss: {val_loss:.6f}")
                    print(f"Error for {param_name}: {errors[0]:.6f}")

                    # Learning rate scheduling
                    scheduler.step(val_loss)

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                        }, os.path.join(param_save_dir, f'best_model_{bound:.2f}.pth'))
                    else:
                        epochs_without_improvement += 1

                    # Early stopping
                    if epochs_without_improvement >= 20:
                        print("Early stopping triggered!")
                        break

                    # Target loss adjusted for larger bounds
                    if val_loss < 0.15:  # Increased from 0.015
                        print(f"Reached target loss for {param_name} at bound {bound}!")
                        break

        # Phase 2: Full coefficient set training
        print("\n=== Training on full coefficient set ===")
        final_save_dir = os.path.join(save_dir, 'final_training')
        os.makedirs(final_save_dir, exist_ok=True)

        try:
            train_loader, val_loader = create_curriculum_loader(
                'TrainingEverything',
                list(range(8)),
                batch_size=32
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error loading TrainingEverything data: {e}")
            return

        # Initialize optimizer and scheduler for final phase
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # Final training loop
        for epoch in range(200):
            print(f"\nFinal Phase - Epoch {epoch}")

            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

            # Validate and save visualizations
            val_loss, errors = validate_with_images(
                model,
                val_loader,
                criterion,
                device,
                param_names,
                final_save_dir,
                epoch
            )

            # Print metrics
            print(f"Training loss: {train_loss:.6f}")
            print(f"Validation loss: {val_loss:.6f}")
            print("Coefficient errors:", errors)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(final_save_dir, 'best_model.pth'))
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= 20:
                print("Early stopping triggered!")
                break

    except Exception as e:
        print(f"Unexpected error during training: {e}")
        # Save emergency checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'error': str(e)
        }, os.path.join(save_dir, 'emergency_checkpoint.pth'))
        raise

    print("\nCurriculum training completed!")
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = PretrainedInterferogramNet(pretrained=False).to(device)

    # Load pretrained weights
    weights = models.ResNet50_Weights.DEFAULT
    pretrained_dict = models.resnet50(weights=weights).state_dict()

    # Handle the first conv layer specially
    conv1_weight = pretrained_dict['conv1.weight']
    pretrained_dict['conv1.weight'] = conv1_weight.mean(dim=1, keepdim=True)

    # Load the modified weights
    model.resnet.load_state_dict(pretrained_dict, strict=False)
    print("Loaded pretrained weights")

    save_dir = 'training_outputs'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created save directory: {save_dir}")

    train_curriculum(model, 'Training', device, save_dir)

if __name__ == '__main__':
    main()