import numpy as np
from PIL import Image
import os

def Interferogram_Plot(D, C, B, G, F, J, E, I, pixels, _):
    """
    Generate an interferogram plot with the given parameters.
    Parameters:
    D: Defocus
    C: Tilt(x)
    B: Tilt(y)
    G: Spherical
    F: Coma(y)
    J: Coma(x)
    E: Astig(y)
    I: Astig(x)
    pixels: Resolution (square)
    _: Unused parameter (kept for compatibility)
    """
    respix = pixels
    x = np.linspace(-(respix/2 - 0.5)/(respix/2), (respix/2 - 0.5)/(respix/2), respix)
    y = np.linspace(-(respix/2 - 0.5)/(respix/2), (respix/2 - 0.5)/(respix/2), respix)
    X, Y = np.meshgrid(x, y)

    OPD = (B * X + C * Y +
           D * (X**2 + Y**2) +
           E * (X**2 + 3*Y**2) +
           F * Y * (X**2 + Y**2) +
           G * (X**2 + Y**2)**2 +
           J * X * (X**2 + Y**2) +
           I * (3*X**2 + Y**2))

    phase = 1 - np.abs(0.5 - (OPD - np.floor(OPD)))/0.5
    grey_plot = Image.fromarray((phase * 255).astype(np.uint8))
    return grey_plot

class DataGenerator:
    def __init__(self, resolution=224, images_per_folder=2000):
        self.res = resolution
        self.images_per_folder = images_per_folder
        self.param_names = ['D', 'C', 'B', 'G', 'F', 'J', 'E', 'I']
        self.main_bounds = 5.0
        np.random.seed(42)  # For reproducibility

    def generate_params(self, active_param_idx, other_bounds):
        """Generate parameters with specific bounds for active and other parameters"""
        params = np.zeros(8)
        for i in range(8):
            if i == active_param_idx:
                params[i] = np.round(np.random.uniform(-self.main_bounds, self.main_bounds), 6)
            else:
                params[i] = np.round(np.random.uniform(-other_bounds, other_bounds), 6)
        return params

    def create_folder_structure(self):
        """Create all necessary training folders"""
        # Create bounded coefficient folders
        for param_idx, param_name in enumerate(self.param_names):
            for bound in np.arange(0, 5, 0.5):
                folder_name = f'Training{param_name}{bound:.1f}'.replace('.', '_')
                os.makedirs(folder_name, exist_ok=True)

                print(f"\nGenerating {self.images_per_folder} images for {folder_name}")
                for img in range(self.images_per_folder):
                    if img % 100 == 0:
                        print(f"Progress: {img}/{self.images_per_folder}")

                    params = self.generate_params(param_idx, bound)
                    image = Interferogram_Plot(*params, self.res, None)

                    # Create filename with parameter values
                    base_filename = f'img_D{params[0]:.6f}_C{params[1]:.6f}_B{params[2]:.6f}_G{params[3]:.6f}_' \
                                  f'F{params[4]:.6f}_J{params[5]:.6f}_E{params[6]:.6f}_I{params[7]:.6f}'
                    base_filename = base_filename.replace('-', 'n').replace('.', 'p')
                    write_file = os.path.join(folder_name, f'{base_filename}.jpg')
                    image.save(write_file)

        # Create the "everything" folder with full bounds
        everything_folder = 'TrainingEverything'
        os.makedirs(everything_folder, exist_ok=True)
        total_everything_images = self.images_per_folder * 10  # 10x more images

        print(f"\nGenerating {total_everything_images} images for {everything_folder}")
        for img in range(total_everything_images):
            if img % 100 == 0:
                print(f"Progress: {img}/{total_everything_images}")

            # Generate completely random parameters within [-5, 5]
            params = np.round(np.random.uniform(-self.main_bounds, self.main_bounds, 8), 6)
            image = Interferogram_Plot(*params, self.res, None)

            base_filename = f'img_D{params[0]:.6f}_C{params[1]:.6f}_B{params[2]:.6f}_G{params[3]:.6f}_' \
                          f'F{params[4]:.6f}_J{params[5]:.6f}_E{params[6]:.6f}_I{params[7]:.6f}'
            base_filename = base_filename.replace('-', 'n').replace('.', 'p')
            write_file = os.path.join(everything_folder, f'{base_filename}.jpg')
            image.save(write_file)

def main():
    generator = DataGenerator(resolution=224, images_per_folder=2000)
    generator.create_folder_structure()

if __name__ == '__main__':
    main()