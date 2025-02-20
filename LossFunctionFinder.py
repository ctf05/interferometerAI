import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import seaborn as sns

# Configure visualization style
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

class AdaptiveLossFunction:
    """
    A class that derives and implements an adaptive loss function with flat expected loss.
    The loss function should provide equal expected loss for any prediction within valid range.
    """
    def __init__(self, valid_range=(-1.0, 1.0), num_samples=2000):
        self.valid_range = valid_range
        self.num_samples = num_samples
        self.params = None  # Will store optimized parameters
        self.loss_function = None  # Will store the derived function

    def _baseline_expected_loss(self, pred, targets):
        """Calculate expected loss for a prediction using saturated loss"""
        abs_diffs = np.abs(pred - targets)
        saturated_diffs = np.minimum(abs_diffs, 0.2)
        return np.mean(saturated_diffs)

    def _parametric_loss(self, diff, params):
        """
        Parametric loss function with adjustable shape
        Combines polynomial components to create flexible curve
        """
        abs_diff = np.abs(diff)

        # Extract parameters
        a, b, c, d, e, p1, p2, scale = params

        # Combine polynomial and exponential terms for flexibility
        loss = (
                       a * abs_diff**p1 +
                       b * abs_diff**p2 +
                       c * abs_diff +
                       d * np.tanh(e * abs_diff)
               ) * scale

        return loss

    def _expected_loss_for_function(self, params, pred_points, target_points):
        """Calculate expected loss across all prediction points using parameter-defined function"""
        expected_losses = []

        for pred in pred_points:
            diffs = pred - target_points
            losses = self._parametric_loss(diffs, params)
            expected_losses.append(np.mean(losses))

        return np.array(expected_losses)

    def _optimization_target(self, params, pred_points, target_points, target_value):
        """Objective function to minimize: variance of expected loss across predictions"""
        expected_losses = self._expected_loss_for_function(params, pred_points, target_points)

        # Calculate deviation from target constant value
        deviation = expected_losses - target_value
        mse = np.mean(deviation**2)

        # Add regularization to keep parameters reasonable
        reg = 0.001 * np.sum(np.array(params)**2)

        return mse + reg

    def find_optimal_loss_function(self, target_expected_loss=0.1):
        """Find parameters that create a loss function with flat expected loss"""
        print(f"\n{'='*60}")
        print(f"STARTING OPTIMIZATION FOR FLAT EXPECTED LOSS: target={target_expected_loss}")
        print(f"{'='*60}")

        # Generate prediction and target points
        pred_points = np.linspace(self.valid_range[0], self.valid_range[1], self.num_samples)
        target_points = np.linspace(self.valid_range[0], self.valid_range[1], self.num_samples)
        print(f"Generated {self.num_samples} prediction points and {self.num_samples} target points")
        print(f"Prediction range: [{self.valid_range[0]}, {self.valid_range[1]}]")

        # Initial parameter guess
        initial_params = [0.5, 0.5, 0.2, 0.1, 5.0, 1.5, 2.0, 1.0]
        print(f"Initial parameters: {initial_params}")

        # Define bounds for parameters
        bounds = [
            (-1.0, 1.0),    # a
            (-1.0, 1.0),   # b
            (-1.0, 1.0),   # c
            (-1.0, 1.0),    # d
            (-2.0, 2.0),   # e
            (0.01, 3.0),    # p1 - power term 1
            (0.01, 4.0),    # p2 - power term 2
            (0.01, 5.0)     # scale
        ]
        print(f"Parameter bounds: {bounds}")

        # Callback to log progress
        iterations = [0]
        best_scores = []

        def callback_log(xk, convergence):
            iterations[0] += 1
            score = self._optimization_target(xk, pred_points, target_points, target_expected_loss)
            best_scores.append(score)

            if iterations[0] % 10 == 0:
                # Calculate flatness metrics
                expected_losses = self._expected_loss_for_function(xk, pred_points, target_points)
                flatness_std = np.std(expected_losses)
                flatness_range = np.max(expected_losses) - np.min(expected_losses)
                avg_loss = np.mean(expected_losses)

                print(f"Iteration {iterations[0]:3d}: Score={score:.8f}, "
                      f"Flatness(std)={flatness_std:.8f}, "
                      f"Range={flatness_range:.8f}, "
                      f"AvgLoss={avg_loss:.6f}")

                if iterations[0] % 50 == 0:
                    # Print current parameter values
                    param_str = ", ".join([f"{p:.6f}" for p in xk])
                    print(f"Current parameters: [{param_str}]")

        print(f"\nStarting differential evolution with population={20}, maxiter={200}...")
        print(f"This may take several minutes. Progress updates every 10 iterations.")
        print(f"{'-'*60}")

        # Use differential evolution for global optimization
        result = differential_evolution(
            self._optimization_target,
            bounds,
            args=(pred_points, target_points, target_expected_loss),
            popsize=20,
            maxiter=200,
            tol=1e-6,
            updating='deferred',
            workers=-1,
            callback=callback_log
        )

        print(f"{'-'*60}")
        print(f"Differential evolution completed in {iterations[0]} iterations")
        print(f"Final score: {result.fun:.10f}")
        print(f"Success: {result.success}")

        if not result.success:
            print(f"\nFalling back to local optimization with multiple starting points...")
            # Fall back to local optimization with multiple starting points
            best_result = None
            best_score = float('inf')

            for i in range(5):
                print(f"\nLocal optimization attempt {i+1}/5")
                # Generate random starting point
                random_params = [
                    np.random.uniform(low, high) for low, high in bounds
                ]
                print(f"Random starting parameters: {[round(p, 4) for p in random_params]}")

                # Progress tracking for this attempt
                current_iter = [0]

                def callback_local(xk):
                    current_iter[0] += 1
                    if current_iter[0] % 20 == 0:
                        score = self._optimization_target(xk, pred_points, target_points, target_expected_loss)
                        print(f"  Local iteration {current_iter[0]}: Score={score:.8f}")

                local_result = minimize(
                    self._optimization_target,
                    random_params,
                    args=(pred_points, target_points, target_expected_loss),
                    bounds=bounds,
                    method='L-BFGS-B',
                    callback=callback_local
                )

                print(f"  Completed: Score={local_result.fun:.10f}, Success={local_result.success}")

                if local_result.fun < best_score:
                    best_score = local_result.fun
                    best_result = local_result
                    print(f"  ↳ New best result!")

            result = best_result
            print(f"\nBest local optimization result:")
            print(f"Score: {result.fun:.10f}")
            print(f"Success: {result.success}")

        self.params = result.x

        # Create the optimized loss function
        def adaptive_loss(diff):
            return self._parametric_loss(diff, self.params)

        self.loss_function = adaptive_loss

        # Add edge correction to the loss function
        def adaptive_loss_with_edge_correction(pred, target):
            diff = pred - target
            base_loss = adaptive_loss(diff)

            # Edge correction with quadratic profile
            edge_correction = 0
            center_distance = abs(pred)

            if center_distance >= 0.8 and center_distance <= 1.0:
                # Edge region (either side)
                edge_position = (center_distance - 0.8) / 0.2  # 0 to 1 scale
                edge_correction = -0.00990 * edge_position**2

            # Range penalty for out-of-range predictions
            range_penalty = 0
            if abs(pred) > 1.0:
                out_of_range = abs(pred) - 1.0
                range_penalty = out_of_range**2

            return base_loss + edge_correction + range_penalty

        self.full_loss_function = adaptive_loss_with_edge_correction

        return result

    def visualize_loss_function(self):
        """Visualize the derived loss function and its expected loss curve"""
        if self.loss_function is None:
            raise ValueError("Must find optimal loss function first")

        # Generate prediction and target points
        pred_points = np.linspace(self.valid_range[0] - 0.2, self.valid_range[1] + 0.2, self.num_samples)
        target_points = np.linspace(self.valid_range[0], self.valid_range[1], self.num_samples)

        # Calculate expected losses for each prediction using both functions
        adaptive_expected_losses = []
        saturated_expected_losses = []

        for pred in pred_points:
            # Adaptive loss
            diffs = pred - target_points
            adaptive_losses = self.loss_function(diffs)
            adaptive_expected_losses.append(np.mean(adaptive_losses))

            # Saturated loss (baseline)
            saturated_expected_losses.append(self._baseline_expected_loss(pred, target_points))

        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 16))

        # 1. Plot expected loss curves
        ax = axes[0]
        ax.plot(pred_points, adaptive_expected_losses, 'r-', linewidth=2,
                label='Adaptive Loss (Expected)')
        ax.plot(pred_points, saturated_expected_losses, 'b--', linewidth=2,
                label='Saturated Loss (Expected)')

        # Add edge zone markers
        ax.axvline(x=-1.0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=-0.8, color='black', linestyle=':', alpha=0.7)
        ax.axvline(x=0.8, color='black', linestyle=':', alpha=0.7)

        # Calculate flatness metrics
        valid_mask = (pred_points >= self.valid_range[0]) & (pred_points <= self.valid_range[1])
        valid_adaptive = np.array(adaptive_expected_losses)[valid_mask]

        flatness_std = np.std(valid_adaptive)
        flatness_range = np.max(valid_adaptive) - np.min(valid_adaptive)

        ax.set_title(f'Expected Loss Curves\n'
                     f'Flatness: std={flatness_std:.8f}, range={flatness_range:.8f}')
        ax.set_xlabel('Prediction Value')
        ax.set_ylabel('Expected Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Plot sample loss curves for different targets
        ax = axes[1]

        # Generate error values
        errors = np.linspace(-2.0, 2.0, 1000)

        # Sample different loss functions
        adaptive_loss = self.loss_function(errors)
        saturated_loss = np.minimum(np.abs(errors), 0.2)

        ax.plot(errors, adaptive_loss, 'r-', linewidth=2, label='Adaptive Loss')
        ax.plot(errors, saturated_loss, 'b--', linewidth=2, label='Saturated Loss')

        ax.set_title('Loss Function Shape')
        ax.set_xlabel('Prediction Error (pred - target)')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Print the derived loss function formula with parameters
        a, b, c, d, e, p1, p2, scale = self.params
        formula = (f"Loss(diff) = ({a:.6f} * |diff|^{p1:.6f} + "
                   f"{b:.6f} * |diff|^{p2:.6f} + "
                   f"{c:.6f} * |diff| + "
                   f"{d:.6f} * tanh({e:.6f} * |diff|)) * {scale:.6f}")

        plt.figtext(0.5, 0.01, formula, ha='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig('adaptive_loss_function.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_loss_landscapes(self):
        """Create 2D heatmaps showing loss landscapes"""
        if self.loss_function is None:
            raise ValueError("Must find optimal loss function first")

        # Generate prediction and target points
        predictions = np.linspace(-1.2, 1.2, 500)
        targets = np.linspace(-1.0, 1.0, 500)

        # Create grid for predictions and targets
        P, T = np.meshgrid(predictions, targets)
        diffs = P - T

        # Calculate adaptive loss
        adaptive_loss = np.vectorize(self.loss_function)(diffs)

        # Calculate saturated loss
        saturated_loss = np.minimum(np.abs(diffs), 0.2)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))

        # Determine common color scale max for consistent comparison
        vmax = max(saturated_loss.max(), adaptive_loss.max())

        # 1. Adaptive Loss
        ax = axes[0, 0]
        im = ax.imshow(adaptive_loss, extent=[-1.2, 1.2, -1.0, 1.0], origin='lower',
                       aspect='auto', cmap='viridis', vmax=vmax)
        fig.colorbar(im, ax=ax, label='Loss')
        ax.set_title('Adaptive Loss')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Target')

        # 2. Saturated Loss
        ax = axes[0, 1]
        im = ax.imshow(saturated_loss, extent=[-1.2, 1.2, -1.0, 1.0], origin='lower',
                       aspect='auto', cmap='viridis', vmax=vmax)
        fig.colorbar(im, ax=ax, label='Loss')
        ax.set_title('Saturated Loss (capped at 0.2)')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Target')

        # 3. Expected Loss Comparison
        ax = axes[1, 0]

        # Calculate expected losses
        adaptive_expected = []
        saturated_expected = []

        for pred in predictions:
            # Calculate losses across all targets
            pred_diffs = pred - targets
            adaptive_losses = self.loss_function(pred_diffs)
            saturated_losses = np.minimum(np.abs(pred_diffs), 0.2)

            adaptive_expected.append(np.mean(adaptive_losses))
            saturated_expected.append(np.mean(saturated_losses))

        # Plot expected losses
        ax.plot(predictions, adaptive_expected, 'r-', linewidth=2, label='Adaptive')
        ax.plot(predictions, saturated_expected, 'b--', linewidth=2, label='Saturated')

        # Add vertical lines at boundaries
        ax.axvline(x=-1.0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=-0.8, color='black', linestyle=':', alpha=0.7)
        ax.axvline(x=0.8, color='black', linestyle=':', alpha=0.7)

        ax.set_title('Expected Loss Comparison')
        ax.set_xlabel('Prediction Value')
        ax.set_ylabel('Expected Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Single example loss curves
        ax = axes[1, 1]

        # Target value for demonstration
        target = 0.0
        pred_values = np.linspace(-1.2, 1.2, 1000)

        # Calculate losses for this target
        adaptive_target_loss = []
        saturated_target_loss = []

        for pred in pred_values:
            diff = pred - target
            adaptive_target_loss.append(self.full_loss_function(pred, target))

            # Calculate saturated loss with edge correction
            sat_loss = min(abs(diff), 0.2)
            # Edge correction
            edge_corr = 0
            if abs(pred) >= 0.8 and abs(pred) <= 1.0:
                edge_position = (abs(pred) - 0.8) / 0.2
                edge_corr = -0.00990 * edge_position**2
            # Range penalty
            range_pen = 0
            if abs(pred) > 1.0:
                range_pen = (abs(pred) - 1.0)**2

            saturated_target_loss.append(sat_loss + edge_corr + range_pen)

        # Plot losses
        ax.plot(pred_values, adaptive_target_loss, 'r-', linewidth=2,
                label='Adaptive Loss with Edge Correction')
        ax.plot(pred_values, saturated_target_loss, 'b--', linewidth=2,
                label='Saturated Loss with Edge Correction')

        # Add target marker
        ax.axvline(x=target, color='k', linestyle='-', alpha=0.5, label='Target')

        # Add boundary markers
        ax.axvline(x=-1.0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=-0.8, color='black', linestyle=':', alpha=0.7)
        ax.axvline(x=0.8, color='black', linestyle=':', alpha=0.7)

        ax.set_title(f'Loss Curves for Target = {target}')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Print derived loss function
        a, b, c, d, e, p1, p2, scale = self.params
        formula = (f"Loss(diff) = ({a:.4f} * |diff|^{p1:.4f} + "
                   f"{b:.4f} * |diff|^{p2:.4f} + "
                   f"{c:.4f} * |diff| + "
                   f"{d:.4f} * tanh({e:.4f} * |diff|)) * {scale:.4f}")

        plt.figtext(0.5, 0.01, formula, ha='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig('adaptive_loss_landscapes.png', dpi=300, bbox_inches='tight')
        plt.close()

    def get_simplified_formula(self):
        """Return a simplified version of the loss function for implementation"""
        if self.params is None:
            raise ValueError("Must find optimal loss function first")

        a, b, c, d, e, p1, p2, scale = self.params

        # Round parameters for readability
        a = round(a, 6)
        b = round(b, 6)
        c = round(c, 6)
        d = round(d, 6)
        e = round(e, 6)
        p1 = round(p1, 6)
        p2 = round(p2, 6)
        scale = round(scale, 6)

        formula = f"""
def adaptive_loss(pred, target):
    # Calculate base loss
    diff = abs(pred - target)
    loss = ({a} * diff**{p1} + {b} * diff**{p2} + {c} * diff + {d} * torch.tanh({e} * diff)) * {scale}
    
    # Apply edge correction
    edge_correction = 0
    center_distance = abs(pred)
    if center_distance >= 0.8 and center_distance <= 1.0:
        # Edge region (either side)
        edge_position = (center_distance - 0.8) / 0.2  # 0 to 1 scale
        edge_correction = -0.00990 * edge_position**2
        
    # Range penalty for out-of-range predictions
    range_penalty = 0
    if abs(pred) > 1.0:
        out_of_range = abs(pred) - 1.0
        range_penalty = out_of_range**2
        
    return loss + edge_correction + range_penalty
"""
        return formula

def derive_adaptive_loss(target_expected_loss=0.1):
    """Main function to derive and visualize the adaptive loss function"""
    import time
    from datetime import datetime

    # Create timestamp for logs and files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"adaptive_loss_optimization_{timestamp}.log"

    # Create adaptive loss finder
    print(f"\n{'-'*80}")
    print(f"ADAPTIVE LOSS FUNCTION OPTIMIZER - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target expected loss: {target_expected_loss}")
    print(f"Log file: {log_file}")
    print(f"{'-'*80}")

    # Tee output to log file
    import sys
    original_stdout = sys.stdout
    log_handle = open(log_file, 'w')

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(original_stdout, log_handle)

    try:
        # Start timer
        start_time = time.time()

        # Initialize finder
        print(f"Initializing AdaptiveLossFunction with valid_range=(-1.0, 1.0), num_samples=2000")
        loss_finder = AdaptiveLossFunction(valid_range=(-1.0, 1.0), num_samples=2000)

        # Find optimal parameters
        result = loss_finder.find_optimal_loss_function(target_expected_loss)

        elapsed = time.time() - start_time
        print(f"\n{'-'*80}")
        print(f"OPTIMIZATION COMPLETE - Time elapsed: {elapsed:.2f} seconds")
        print(f"Optimization result: {'SUCCESS' if result.success else 'FAILURE'}")
        print(f"Final score: {result.fun:.10f}")

        # Print each parameter with its meaning
        a, b, c, d, e, p1, p2, scale = loss_finder.params
        params_desc = [
            ("a (power term 1 coef)", a),
            ("b (power term 2 coef)", b),
            ("c (linear term coef)", c),
            ("d (tanh term coef)", d),
            ("e (tanh scaling)", e),
            ("p1 (power 1 exponent)", p1),
            ("p2 (power 2 exponent)", p2),
            ("scale (global scaling)", scale)
        ]

        print(f"\nOptimized parameters:")
        for desc, value in params_desc:
            print(f"  {desc.ljust(25)}: {value:.8f}")

        # Evaluate flatness of expected loss
        print(f"\nEvaluating expected loss flatness...")
        pred_points = np.linspace(-1.0, 1.0, 1000)
        target_points = np.linspace(-1.0, 1.0, 1000)
        expected_losses = loss_finder._expected_loss_for_function(
            loss_finder.params, pred_points, target_points)

        flatness_stats = {
            "mean": np.mean(expected_losses),
            "std": np.std(expected_losses),
            "min": np.min(expected_losses),
            "max": np.max(expected_losses),
            "range": np.max(expected_losses) - np.min(expected_losses)
        }

        print(f"Expected loss statistics:")
        print(f"  Mean:  {flatness_stats['mean']:.8f}")
        print(f"  Std:   {flatness_stats['std']:.8f}")
        print(f"  Min:   {flatness_stats['min']:.8f}")
        print(f"  Max:   {flatness_stats['max']:.8f}")
        print(f"  Range: {flatness_stats['range']:.8f}")

        # Create visualization filenames with timestamp
        vis_file1 = f"adaptive_loss_function_{timestamp}.png"
        vis_file2 = f"adaptive_loss_landscapes_{timestamp}.png"
        formula_file = f"adaptive_loss_formula_{timestamp}.py"

        # Visualize the results
        print(f"\nCreating visualizations...")
        print(f"  - Function visualization: {vis_file1}")
        print(f"  - Landscapes visualization: {vis_file2}")
        loss_finder.visualize_loss_function()
        loss_finder.visualize_loss_landscapes()

        # Get simplified formula
        formula = loss_finder.get_simplified_formula()
        print(f"\nDerived Loss Function:")
        print(formula)

        # Save the formula to a file
        with open(formula_file, 'w') as f:
            f.write(f"# Adaptive Loss Function derived on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Flatness metrics: mean={flatness_stats['mean']:.8f}, std={flatness_stats['std']:.8f}\n")
            f.write(f"# Parameter values: {[round(p, 8) for p in loss_finder.params]}\n\n")
            f.write(formula)

        print(f"\nFormula saved to '{formula_file}'")
        print(f"{'-'*80}")
        print(f"PROCESS COMPLETE - Total time: {time.time() - start_time:.2f} seconds")

        return loss_finder

    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_handle.close()
        print(f"Log saved to {log_file}")

if __name__ == "__main__":
    # Target expected loss similar to saturated loss
    derive_adaptive_loss(target_expected_loss=0.1)