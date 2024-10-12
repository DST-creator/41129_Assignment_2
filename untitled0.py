import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# Function to create the custom marker
def triangle_down_center_mark(size=10):
    # Coordinates for the equilateral triangle (pointing down)
    triangle = np.array([
        [-0.35, .15],   # top left corner
        [0.35, .15],    # top right corner
        [0.0, -.5],   # bottom corner
        [-0.35, .15]    # closing the path
    ]) * size

    # Calculate the center of the triangle
    center = np.array([0, -np.sqrt(3)/9]) * size

    # Vertices for lines from corners to the center of the triangle
    vertices = np.concatenate([
        triangle,            # triangle outline
        [triangle[0], center, triangle[1], center, triangle[2], center]  # lines to center
    ])

    # Path codes to specify the drawing order
    codes = [
        Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,  # triangle
        Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO  # lines from corners
    ]

    # Create and return the custom Path object
    return Path(vertices, codes)
