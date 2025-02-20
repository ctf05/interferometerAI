import numpy as np
from PIL import Image
import os

def Interferogram_Plot(D, C, B, G, F, J, E, I, pixels, _):
    C = 0
    B = 0
    G = 0
    F = 0
    J = 0
    E = 0
    I = 0
    """
    Generate an interferogram plot with the given parameters.

    Parameters same as MATLAB version:
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

    WFE = np.zeros((respix, respix))

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

    WFE = OPD

    phase = 1 - np.abs(0.5 - (WFE - np.floor(WFE)))/0.5

    grey_plot = Image.fromarray((phase * 255).astype(np.uint8))

    return grey_plot

cap = 5
num_images = 10000
res = 224

if not os.path.exists('training'):
    os.makedirs('training')

np.random.seed()

for img in range(num_images):
    params = np.round(np.random.uniform(-cap, cap, 8), 6)

    image = Interferogram_Plot(params[0], params[1], params[2], params[3],
                               params[4], params[5], params[6], params[7],
                               res, None)

    base_filename = f'img_D{params[0]:.6f}_C{params[1]:.6f}_B{params[2]:.6f}_G{params[3]:.6f}_' \
                    f'F{params[4]:.6f}_J{params[5]:.6f}_E{params[6]:.6f}_I{params[7]:.6f}'

    base_filename = base_filename.replace('-', 'n').replace('.', 'p')

    write_file = os.path.join('training', f'{base_filename}.jpg')
    image.save(write_file)