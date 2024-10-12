import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle

# Function to create the custom marker
def triangle_down_center_mark(size=10):
    # Coordinates for the equilateral triangle (pointing down)
    triangle = np.array([
        [-0.35, .15],   # top left corner
        [0.35, .15],    # top right corner
        [0.0, -.6],   # bottom corner
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

# Function to create the custom marker
def diamond_center_mark(size=10):
    # Coordinates for the equilateral triangle (pointing down)
    diamond = np.array([
        [-0.3, 0],   # left corner
        [0, .5],    # top  corner
        [0.3, 0],   # right corner
        [0, -.5],   # bottom corner
        [-0.3, 0]    # closing the path
    ]) * size

    # Calculate the center of the triangle
    center = np.array([0, 0]) * size

    # Vertices for lines from corners to the center of the triangle
    vertices = np.concatenate([
        diamond,            # triangle outline
        [diamond[0], diamond[2], diamond[1], diamond[3]]  # lines to center
    ])

    # Path codes to specify the drawing order
    codes = [
        Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,  # diamond
        Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO  # lines for center mark
    ]

    # Create and return the custom Path object
    return Path(vertices, codes)
