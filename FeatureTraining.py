import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import re
import glob
from pathlib import Path
import cv2
import scipy.ndimage as ndimage
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.special import factorial

def unwrap_phase_custom(phase):
    """
    Simple phase unwrapping using a 2D gradient-based approach.
    """
    return np.unwrap(np.unwrap(phase, axis=0), axis=1)

def extract_phase_gradient(interferogram):
    """
    Compute phase gradients using Fourier transform method.
    """
    # Convert image to grayscale if necessary
    if len(interferogram.shape) == 3:
        interferogram = cv2.cvtColor(interferogram, cv2.COLOR_BGR2GRAY)

    # Apply Fourier Transform
    f_transform = fft2(interferogram)
    f_transform_shifted = fftshift(f_transform)

    # Compute phase
    phase = np.angle(f_transform_shifted)

    # Unwrap phase
    unwrapped_phase = unwrap_phase_custom(phase)

    return unwrapped_phase

def process_images(image_dir):
    """
    Process all images in a directory and extract phase gradients.
    """
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    phase_maps = []

    for i, image_path in enumerate(image_paths):
        if i % 50 == 0:
            print(f"Processing image {i}/{len(image_paths)}")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            phase_map = extract_phase_gradient(image)
            phase_maps.append(phase_map)

    return np.array(phase_maps)

def train_model(X, y, model_type="svr"):
    """
    Train a regression model (SVR or MLP) based on extracted features.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "svr":
        model = SVR(kernel="rbf")
    elif model_type == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {model_type}, MSE: {mse:.4f}, R2: {r2:.4f}")

    return model, scaler

# Example usage
if __name__ == "__main__":
    image_directory = "./TrainingEverything"
    phase_data = process_images(image_directory)

    # Dummy labels for demonstration
    labels = np.random.rand(len(phase_data))

    trained_model, trained_scaler = train_model(phase_data.reshape(len(phase_data), -1), labels, model_type="svr")

    # Example predictions
    example_data = phase_data[:5].reshape(50, -1)  # Take the first 5 samples for prediction
    example_data_scaled = trained_scaler.transform(example_data)
    predictions = trained_model.predict(example_data_scaled)

    print("\nExample Predictions and Actual Labels:")
    for i, (prediction, actual) in enumerate(zip(predictions, labels[:50])):
        print(f"Sample {i+1}: Prediction: {prediction:.4f}, Actual: {actual:.4f}")