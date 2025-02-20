import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set a nice aesthetic style
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

def calculate_loss_landscapes():
    # Generate prediction and target points
    predictions = np.linspace(-1.2, 1.2, 500)  # Include out-of-range
    targets = np.linspace(-1.0, 1.0, 500)

    # Create grid for predictions and targets
    P, T = np.meshgrid(predictions, targets)

    # Calculate different losses
    # 1. MSE loss
    mse_loss = (P - T)**2

    # 2. Saturated loss (capped at 0.2)
    diffs = np.abs(P - T)
    saturated_loss = np.minimum(diffs, 0.2)

    # 3. Saturated loss with edge correction
    # Initialize with saturated loss
    edge_corrected_loss = saturated_loss.copy()

    # Apply edge correction
    for i, p in enumerate(predictions):
        # Initialize edge correction for this prediction
        edge_correction = np.zeros_like(targets)

        # Apply correction with quadratic profile in edge regions
        if p <= -0.8 and p >= -1.0:
            # Left edge region
            edge_position = (-p - 0.8) / 0.2  # 0 at -0.8, 1 at -1.0
            edge_correction = -0.00990 * edge_position**2  # Quadratic profile
        elif p >= 0.8 and p <= 1.0:
            # Right edge region
            edge_position = (p - 0.8) / 0.2  # 0 at 0.8, 1 at 1.0
            edge_correction = -0.00990 * edge_position**2  # Quadratic profile

        # Add out-of-range penalty if needed
        if abs(p) > 1.0:
            out_of_range = abs(p) - 1.0
            range_penalty = out_of_range**2
            edge_correction += range_penalty

        # Apply correction
        edge_corrected_loss[:, i] += edge_correction

    return predictions, targets, mse_loss, saturated_loss, edge_corrected_loss

def calculate_expected_losses(num_points=1000, num_targets=1000):
    # Generate prediction points across the range, including out-of-range
    pred_points = np.linspace(-1.2, 1.2, num_points)

    # Generate target points within valid range
    target_points = np.linspace(-1.0, 1.0, num_targets)

    # Initialize arrays for loss values
    mse_expected_loss = np.zeros_like(pred_points)
    saturated_expected_loss = np.zeros_like(pred_points)
    edge_corrected_expected_loss = np.zeros_like(pred_points)
    edge_correction_values = np.zeros_like(pred_points)
    range_penalty_values = np.zeros_like(pred_points)

    # Calculate expected losses for each prediction point
    for i, pred in enumerate(pred_points):
        # Calculate losses across all possible targets
        diffs = pred - target_points
        abs_diffs = np.abs(diffs)

        # 1. MSE loss
        mse_losses = diffs**2
        mse_expected_loss[i] = np.mean(mse_losses)

        # 2. Saturated loss
        saturated_diffs = np.minimum(abs_diffs, 0.2)
        saturated_expected_loss[i] = np.mean(saturated_diffs)

        # 3. Edge correction with quadratic profile
        edge_correction = 0
        center_distance = abs(pred)
        if center_distance >= 0.8 and center_distance <= 1.0:
            # Edge region (either side)
            edge_position = (center_distance - 0.8) / 0.2  # 0 at 0.8, 1 at 1.0
            edge_correction = -0.00990 * edge_position**2  # Quadratic profile

        # 4. Range penalty for out-of-range predictions
        range_penalty = 0
        if abs(pred) > 1.0:
            out_of_range = abs(pred) - 1.0
            range_penalty = out_of_range**2

        edge_correction_values[i] = edge_correction
        range_penalty_values[i] = range_penalty
        edge_corrected_expected_loss[i] = saturated_expected_loss[i] + edge_correction + range_penalty

    return (pred_points, mse_expected_loss, saturated_expected_loss,
            edge_corrected_expected_loss, edge_correction_values, range_penalty_values)

def visualize_loss_landscapes():
    """Create 2D heatmaps showing loss landscapes for different loss functions
    Now with quadratic edge correction profile"""

    predictions, targets, mse_loss, saturated_loss, edge_corrected_loss = calculate_loss_landscapes()

    # Create figure
    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 0.8], width_ratios=[1, 0.05])

    # Determine common color scale max for consistent comparison
    vmax = max(saturated_loss.max(), edge_corrected_loss.max())

    # 1. MSE Loss
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(mse_loss, extent=[-1.2, 1.2, -1.0, 1.0], origin='lower',
                     aspect='auto', cmap='viridis')
    ax1.set_title('MSE Loss')
    ax1.set_xlabel('Prediction')
    ax1.set_ylabel('Target')
    # Add colorbar
    cax1 = fig.add_subplot(gs[0, 1])
    fig.colorbar(im1, cax=cax1, label='Loss')

    # 2. Saturated Loss
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(saturated_loss, extent=[-1.2, 1.2, -1.0, 1.0], origin='lower',
                     aspect='auto', cmap='viridis', vmax=vmax)
    ax2.set_title('Saturated Loss (capped at 0.2)')
    ax2.set_xlabel('Prediction')
    ax2.set_ylabel('Target')
    # Add colorbar
    cax2 = fig.add_subplot(gs[1, 1])
    fig.colorbar(im2, cax=cax2, label='Loss')

    # 3. Edge-corrected Saturated Loss
    ax3 = fig.add_subplot(gs[2, 0])
    im3 = ax3.imshow(edge_corrected_loss, extent=[-1.2, 1.2, -1.0, 1.0], origin='lower',
                     aspect='auto', cmap='viridis', vmax=vmax)
    ax3.set_title('Edge-corrected Saturated Loss (Quadratic Profile)')
    ax3.set_xlabel('Prediction')
    ax3.set_ylabel('Target')
    # Add colorbar
    cax3 = fig.add_subplot(gs[2, 1])
    fig.colorbar(im3, cax=cax3, label='Loss')

    # Add boundary markers
    for ax in [ax1, ax2, ax3]:
        ax.axvline(x=-1.0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=-0.8, color='black', linestyle=':', alpha=0.7)
        ax.axvline(x=0.8, color='black', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('loss_landscapes.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_expected_losses():
    """Create plots of expected loss across prediction range
    With quadratic edge correction profile to avoid local minima"""

    # Calculate expected losses
    results = calculate_expected_losses(num_points=1000)
    pred_points, mse_expected, saturated_expected, edge_corrected_expected, edge_correction, range_penalty = results

    # Create main figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1])

    # 1. All expected losses comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(pred_points, mse_expected, 'b-', linewidth=2, label='MSE Loss')
    ax1.plot(pred_points, saturated_expected, 'g-', linewidth=2, label='Saturated Loss')
    ax1.plot(pred_points, edge_corrected_expected, 'r-', linewidth=2, label='Edge-corrected Loss')

    # Add vertical lines at boundaries
    ax1.axvline(x=-1.0, color='red', linestyle='--', alpha=0.7, label='Valid Range Boundary')
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    ax1.axvline(x=-0.8, color='black', linestyle=':', alpha=0.7, label='Edge Correction Zone')
    ax1.axvline(x=0.8, color='black', linestyle=':', alpha=0.7)

    ax1.set_title('Expected Loss Comparison Across Prediction Range')
    ax1.set_xlabel('Prediction Value')
    ax1.set_ylabel('Expected Loss')
    ax1.legend(loc='upper center')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1.2, 1.2)

    # 2. Zoomed in view of saturated loss
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(pred_points, saturated_expected, 'g-', linewidth=2)

    # Calculate and highlight values at key points
    center_idx = np.argmin(np.abs(pred_points))
    edge_idx = np.argmin(np.abs(pred_points - 1.0))
    neg_edge_idx = np.argmin(np.abs(pred_points + 1.0))

    # Highlight key points
    ax2.plot(pred_points[center_idx], saturated_expected[center_idx], 'ro', markersize=8,
             label=f'Center: {saturated_expected[center_idx]:.5f}')
    ax2.plot(pred_points[edge_idx], saturated_expected[edge_idx], 'bo', markersize=8,
             label=f'Edge (1.0): {saturated_expected[edge_idx]:.5f}')
    ax2.plot(pred_points[neg_edge_idx], saturated_expected[neg_edge_idx], 'mo', markersize=8,
             label=f'Edge (-1.0): {saturated_expected[neg_edge_idx]:.5f}')

    # Add horizontal lines showing the difference
    ax2.axhline(y=saturated_expected[center_idx], color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=saturated_expected[edge_idx], color='b', linestyle='--', alpha=0.5)

    # Calculate and display the difference
    diff = saturated_expected[edge_idx] - saturated_expected[center_idx]
    ax2.annotate(f'Difference: {diff:.5f}',
                 xy=(0.5, (saturated_expected[edge_idx] + saturated_expected[center_idx])/2),
                 xycoords='data', ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    ax2.set_title('Original Saturated Loss - Edge Disadvantage')
    ax2.set_xlabel('Prediction Value')
    ax2.set_ylabel('Expected Loss')
    ax2.legend(loc='upper center')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.1, 1.1)

    # 3. Corrected loss with components
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(pred_points, edge_corrected_expected, 'r-', linewidth=2, label='Corrected Loss')
    ax3.plot(pred_points, saturated_expected, 'g--', linewidth=1.5, alpha=0.7, label='Original Loss')

    # Add edge correction and range penalty components
    valid_range_mask = (pred_points >= -1.0) & (pred_points <= 1.0)
    ax3.plot(pred_points[valid_range_mask], edge_correction[valid_range_mask], 'b-', linewidth=1.5,
             alpha=0.7, label='Edge Correction')
    ax3.plot(pred_points, range_penalty, 'm-', linewidth=1.5, alpha=0.7, label='Range Penalty')

    # Highlight key points
    ax3.plot(pred_points[center_idx], edge_corrected_expected[center_idx], 'ro', markersize=8,
             label=f'Center: {edge_corrected_expected[center_idx]:.5f}')
    ax3.plot(pred_points[edge_idx], edge_corrected_expected[edge_idx], 'bo', markersize=8,
             label=f'Edge (1.0): {edge_corrected_expected[edge_idx]:.5f}')

    ax3.set_title('Edge-corrected Loss with Quadratic Correction')
    ax3.set_xlabel('Prediction Value')
    ax3.set_ylabel('Loss')
    ax3.legend(loc='upper center')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1.1, 1.1)

    # 4. Detailed view of edge correction regions
    ax4 = fig.add_subplot(gs[2, :])

    # Filter to just valid range
    valid_idx = (pred_points >= -1.0) & (pred_points <= 1.0)
    valid_preds = pred_points[valid_idx]
    valid_saturated = saturated_expected[valid_idx]
    valid_corrected = edge_corrected_expected[valid_idx]
    valid_edge_correction = edge_correction[valid_idx]

    # Create shaded regions for edge correction zones
    left_edge_zone = (valid_preds >= -1.0) & (valid_preds <= -0.8)
    right_edge_zone = (valid_preds >= 0.8) & (valid_preds <= 1.0)
    center_zone = (valid_preds > -0.8) & (valid_preds < 0.8)

    # Plot with shaded regions
    ax4.fill_between(valid_preds[left_edge_zone], 0, valid_saturated[left_edge_zone],
                     color='blue', alpha=0.2, label='Left Edge Zone')
    ax4.fill_between(valid_preds[right_edge_zone], 0, valid_saturated[right_edge_zone],
                     color='green', alpha=0.2, label='Right Edge Zone')
    ax4.fill_between(valid_preds[center_zone], 0, valid_saturated[center_zone],
                     color='gray', alpha=0.1, label='Central Zone (No Correction)')

    # Plot the losses
    ax4.plot(valid_preds, valid_saturated, 'b-', linewidth=2, label='Original Saturated Loss')
    ax4.plot(valid_preds, valid_corrected, 'r-', linewidth=2, label='Edge-corrected Loss')
    ax4.plot(valid_preds, valid_edge_correction, 'g-', linewidth=1.5,
             label='Edge Correction Component')

    # Add horizontal guideline at center expected loss
    center_idx_valid = np.argmin(np.abs(valid_preds))
    ax4.axhline(y=valid_corrected[center_idx_valid], color='black', linestyle='--', alpha=0.5,
                label=f'Target Equal Loss: {valid_corrected[center_idx_valid]:.5f}')

    # Annotations for key points
    left_edge_idx = np.argmin(np.abs(valid_preds + 1.0))
    right_edge_idx = np.argmin(np.abs(valid_preds - 1.0))

    ax4.annotate(f'Corrected: {valid_corrected[left_edge_idx]:.5f}',
                 xy=(valid_preds[left_edge_idx], valid_corrected[left_edge_idx]),
                 xytext=(-0.9, valid_corrected[left_edge_idx] + 0.005),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red'),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    ax4.annotate(f'Corrected: {valid_corrected[right_edge_idx]:.5f}',
                 xy=(valid_preds[right_edge_idx], valid_corrected[right_edge_idx]),
                 xytext=(0.9, valid_corrected[right_edge_idx] + 0.005),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red'),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    ax4.set_title('Detailed View of Quadratic Edge Correction (Valid Range Only)')
    ax4.set_xlabel('Prediction Value')
    ax4.set_ylabel('Expected Loss')
    ax4.legend(loc='upper center', ncol=3)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1.0, 1.0)

    plt.tight_layout()
    plt.savefig('expected_loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_sample_loss_curves():
    """Create plots showing loss curves for specific targets
    Using quadratic edge correction profile"""

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Generate prediction points
    predictions = np.linspace(-1.2, 1.2, 1000)

    # Sample targets to visualize
    targets = [-0.8, -0.3, 0.3, 0.8]

    for i, target in enumerate(targets):
        ax = axes[i//2, i%2]

        # Calculate losses
        mse_loss = (predictions - target)**2
        abs_diff = np.abs(predictions - target)
        saturated_loss = np.minimum(abs_diff, 0.2)

        # Calculate edge correction with quadratic profile
        edge_correction = np.zeros_like(predictions)
        for j, pred in enumerate(predictions):
            # Use center distance for symmetric treatment
            center_distance = abs(pred)
            if center_distance >= 0.8 and center_distance <= 1.0:
                # Edge region (either side)
                edge_position = (center_distance - 0.8) / 0.2  # 0 at 0.8, 1 at 1.0
                edge_correction[j] = -0.00990 * edge_position**2  # Quadratic profile

        # Range penalty
        range_penalty = np.zeros_like(predictions)
        out_of_range = np.abs(predictions) > 1.0
        range_penalty[out_of_range] = (np.abs(predictions[out_of_range]) - 1.0)**2

        # Combined edge-corrected loss
        edge_corrected_loss = saturated_loss + edge_correction + range_penalty

        # Plot the losses
        ax.plot(predictions, mse_loss, 'b-', linewidth=2, alpha=0.7, label='MSE Loss')
        ax.plot(predictions, saturated_loss, 'g-', linewidth=2, label='Saturated Loss')
        ax.plot(predictions, edge_corrected_loss, 'r-', linewidth=2, label='Edge-corrected Loss')
        ax.plot(predictions, edge_correction, 'c--', linewidth=1.5, alpha=0.7, label='Edge Correction')
        ax.plot(predictions, range_penalty, 'm--', linewidth=1.5, alpha=0.7, label='Range Penalty')

        # Add vertical lines
        ax.axvline(x=target, color='black', linestyle='-', alpha=0.5, label='Target')
        ax.axvline(x=-1.0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=-0.8, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5)

        # Set title and labels
        ax.set_title(f'Loss Curves for Target = {target}')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.2, 1.2)

        # Zoom in to show the subtle edge correction
        y_max = 0.3 if i == 0 else 0.5  # Adjust based on target
        ax.set_ylim(0, y_max)

    plt.tight_layout()
    plt.savefig('sample_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_final_comparison():
    """Create a focused comparison of original vs corrected expected loss
    Using quadratic edge correction profile to ensure uniform expected loss"""

    # Calculate expected losses
    results = calculate_expected_losses(num_points=2000, num_targets=2000)
    pred_points, _, saturated_expected, edge_corrected_expected, _, _ = results

    # Create figure
    plt.figure(figsize=(14, 8))

    # Filter to valid range
    valid_idx = (pred_points >= -1.0) & (pred_points <= 1.0)
    valid_preds = pred_points[valid_idx]
    valid_saturated = saturated_expected[valid_idx]
    valid_corrected = edge_corrected_expected[valid_idx]

    # Calculate statistics
    center_idx = np.argmin(np.abs(valid_preds))
    center_value = valid_preds[center_idx]
    left_edge_idx = np.argmin(np.abs(valid_preds + 1.0))
    right_edge_idx = np.argmin(np.abs(valid_preds - 1.0))

    # Original loss stats
    original_center = valid_saturated[center_idx]
    original_left = valid_saturated[left_edge_idx]
    original_right = valid_saturated[right_edge_idx]
    original_diff_left = original_left - original_center
    original_diff_right = original_right - original_center

    # Corrected loss stats
    corrected_center = valid_corrected[center_idx]
    corrected_left = valid_corrected[left_edge_idx]
    corrected_right = valid_corrected[right_edge_idx]
    corrected_diff_left = corrected_left - corrected_center
    corrected_diff_right = corrected_right - corrected_center

    # Plot original expected loss
    plt.plot(valid_preds, valid_saturated, 'b-', linewidth=3, alpha=0.7,
             label='Original Saturated Loss')

    # Plot corrected expected loss
    plt.plot(valid_preds, valid_corrected, 'r-', linewidth=3,
             label='Edge-corrected Loss')

    # Add horizontal lines for original loss values
    plt.axhline(y=original_center, color='blue', linestyle='--', alpha=0.5,
                label=f'Original Center: {original_center:.5f}')
    plt.axhline(y=original_left, color='blue', linestyle=':', alpha=0.5)
    plt.axhline(y=original_right, color='blue', linestyle=':', alpha=0.5)

    # Add horizontal line for corrected loss (should be uniform)
    plt.axhline(y=corrected_center, color='red', linestyle='--', alpha=0.5,
                label=f'Corrected (Equal): {corrected_center:.5f}')

    # Add edge zone markers
    plt.axvline(x=-0.8, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=0.8, color='gray', linestyle=':', alpha=0.7)

    # Add annotations
    plt.annotate(f'Original Diff: {original_diff_left:.5f}',
                 xy=(-0.9, (original_left + original_center)/2),
                 xytext=(-0.6, (original_left + original_center)/2 + 0.005),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue'),
                 color='blue',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    plt.annotate(f'Original Diff: {original_diff_right:.5f}',
                 xy=(0.9, (original_right + original_center)/2),
                 xytext=(0.6, (original_right + original_center)/2 + 0.005),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue'),
                 color='blue',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    plt.annotate(f'Corrected Diff: {corrected_diff_left:.6f}',
                 xy=(-0.9, corrected_left),
                 xytext=(-0.5, corrected_left - 0.01),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red'),
                 color='red',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    plt.annotate(f'Corrected Diff: {corrected_diff_right:.6f}',
                 xy=(0.9, corrected_right),
                 xytext=(0.5, corrected_right - 0.01),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red'),
                 color='red',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # Add title and labels
    plt.title('Comparison of Expected Loss: Original vs. Quadratic Edge-corrected', fontsize=16)
    plt.xlabel('Prediction Value', fontsize=14)
    plt.ylabel('Expected Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(-1.0, 1.0)

    # Add summary text
    summary_text = (
        f"Original Loss: Center={original_center:.5f}, Edges={original_left:.5f}/{original_right:.5f}\n"
        f"Edge Disadvantage: {original_diff_left:.5f}/{original_diff_right:.5f}\n\n"
        f"Corrected Loss: Center={corrected_center:.5f}, Edges={corrected_left:.5f}/{corrected_right:.5f}\n"
        f"Remaining Difference: {corrected_diff_left:.6f}/{corrected_diff_right:.6f}"
    )
    plt.annotate(summary_text, xy=(0.02, 0.02), xycoords='figure fraction',
                 fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_all_visualizations():
    """Generate all visualizations"""
    visualize_loss_landscapes()
    visualize_expected_losses()
    visualize_sample_loss_curves()
    visualize_final_comparison()
    print("All visualizations created successfully.")

if __name__ == "__main__":
    create_all_visualizations()