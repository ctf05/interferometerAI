import os
import re
import imageio
import natsort
from PIL import Image
import numpy as np

def create_gif_from_plots(directory, output_path='training_visualization.gif', duration=0.5):
    """
    Create a GIF from matplotlib plots in a directory, resizing all images to the same dimensions.
    """
    # Get all image files
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort files naturally
    # First, separate files into different categories
    warmup_files = [f for f in image_files if 'warmup' in f.lower()]
    final_phase_files = [f for f in image_files if 'final_phase' in f.lower()]

    # Sort other files (coefficient-specific plots)
    coefficient_files = [f for f in image_files if any(letter in f.lower() for letter in 'dcbgjei')]

    # Custom sorting function
    def sort_key(filename):
        # Extract epoch number
        epoch_match = re.search(r'epoch_?(\d+)', filename)
        epoch = int(epoch_match.group(1)) if epoch_match else 0

        # Prioritize order of letters
        letter_order = 'DCBGJEI'
        letter_match = re.search(r'([DCBGJEI])', filename, re.IGNORECASE)
        letter_priority = letter_order.index(letter_match.group(1).upper()) if letter_match else len(letter_order)

        return (letter_priority, epoch)

    # Sort the files
    warmup_files = natsort.natsorted(warmup_files)
    final_phase_files = natsort.natsorted(final_phase_files)
    coefficient_files = sorted(coefficient_files, key=sort_key)

    # Combine the lists in order
    all_files = warmup_files + coefficient_files

    # Full paths to image files
    image_paths = [os.path.join(directory, f) for f in all_files]

    # First, determine the target size by getting the most common image size
    sizes = {}
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            size = img.size
            sizes[size] = sizes.get(size, 0) + 1
        except Exception as e:
            print(f"Could not check size of {image_path}: {e}")

    # Find the most common size
    if sizes:
        target_size = max(sizes.items(), key=lambda x: x[1])[0]
        print(f"Resizing all images to {target_size}")
    else:
        target_size = (1200, 800)  # Default size if no images could be read
        print(f"Using default size: {target_size}")

    # Read and resize images
    images = []
    for image_path in image_paths:
        try:
            # Open with PIL and resize
            with Image.open(image_path) as img:
                img_resized = img.resize(target_size, Image.LANCZOS)
                # Convert to numpy array for imageio
                img_array = np.array(img_resized)
                images.append(img_array)
            print(f"Processed: {image_path}")
        except Exception as e:
            print(f"Could not process {image_path}: {e}")

    if images:
        # Create GIF
        print(f"Creating GIF with {len(images)} images...")
        imageio.v2.mimsave(output_path, images, duration=duration)
        print(f"GIF created at {output_path}")
    else:
        print("No images could be processed. GIF creation failed.")

# Example usage
if __name__ == '__main__':
    # Specify the directory containing your plots
    plot_directory = 'curriculum_training_20250220_120144'  # Replace with your actual directory

    # Create GIF
    create_gif_from_plots(plot_directory)