import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set aesthetic style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

def calculate_new_loss_landscapes():
    """Calculate the new loss landscape and expected losses."""
    predictions = np.linspace(-1.2, 1.2, 500)
    targets = np.linspace(-1.0, 1.0, 500)
    P, T = np.meshgrid(predictions, targets)

    # New loss function: L(P, T) = |P-T| + 0.5*(1 - P²)
    new_loss = np.abs(P - T) + 0.5 * (1 - P**2)

    # Calculate expected losses
    pred_points = np.linspace(-1.2, 1.2, 1000)
    target_points = np.linspace(-1.0, 1.0, 1000)
    new_expected = np.zeros_like(pred_points)
    for i, pred in enumerate(pred_points):
        new_losses = np.abs(pred - target_points) + 0.5 * (1 - pred**2)
        new_expected[i] = np.mean(new_losses)

    return predictions, targets, new_loss, pred_points, new_expected

def visualize_new_loss():
    """Visualize the new loss landscape, expected loss, and sample curves."""
    preds, targs, loss, exp_preds, exp_loss = calculate_new_loss_landscapes()

    # Loss landscape
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(loss, extent=[-1.2, 1.2, -1.0, 1.0], origin='lower',
                    aspect='auto', cmap='viridis')
    ax1.set_title("New Loss Landscape")
    ax1.set_xlabel("Prediction")
    ax1.set_ylabel("Target")
    ax1.axvline(-1.0, color='red', ls='--', alpha=0.7)
    ax1.axvline(1.0, color='red', ls='--', alpha=0.7)
    plt.colorbar(im, ax=ax1, label='Loss')

    # Expected loss
    ax2 = fig.add_subplot(122)
    ax2.plot(exp_preds, exp_loss, 'b-', lw=2)
    ax2.axvline(-1.0, color='red', ls='--', alpha=0.7, label='Valid Range')
    ax2.axvline(1.0, color='red', ls='--', alpha=0.7)
    ax2.set_title("Expected Loss for New Loss Function")
    ax2.set_xlabel("Prediction Value")
    ax2.set_ylabel("Expected Loss")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(0.95, 1.05)

    plt.tight_layout()
    plt.savefig('new_loss_landscape_expected.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Sample loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    targets = [-0.8, -0.3, 0.3, 0.8]
    preds = np.linspace(-1.2, 1.2, 500)
    for i, t in enumerate(targets):
        ax = axes[i//2, i%2]
        loss = np.abs(preds - t) + 0.5 * (1 - preds**2)
        ax.plot(preds, loss, 'r-', lw=2)
        ax.axvline(t, color='k', ls='-', alpha=0.5, label='Target')
        ax.axvline(-1.0, color='red', ls='--', alpha=0.5)
        ax.axvline(1.0, color='red', ls='--', alpha=0.5)
        ax.set_title(f'Target = {t}')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig('new_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_all_visualizations():
    """Generate all new loss visualizations."""
    visualize_new_loss()
    print("Visualizations for new loss created successfully.")

if __name__ == "__main__":
    create_all_visualizations()