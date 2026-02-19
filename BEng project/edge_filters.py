import numpy as np
from scipy.ndimage import sobel, binary_opening, binary_closing



def sobel_mag(img, mask=None):
    """Sobel gradient magnitude. If mask is given, zeros outside mask."""

    # Make a copy so we don't change the original image
    x = img.copy()

    # If mask, set everything outside it to zero
    if mask is not None:
        x = x.astype(np.float32)
        # Find pixels NOT in the mask
        not_mask = (mask == False)
        x[not_mask] = 0.0

    # Calculate horizontal and vertical edges
    gx = sobel(x, axis=1)  # Horizontal changes
    gy = sobel(x, axis=0)  # Vertical changes

    # Combine them to get overall edge strength
    g = np.sqrt(gx * gx + gy * gy)

    # If mask, background is zero
    if mask is not None:
        not_mask = (mask == False)
        g[not_mask] = 0.0

    return g


def clean_binary_edge(edge_bin, k=5):
    """Light cleanup to remove speckle and connect small gaps."""

    # Create a 3x3 square of ones
    structure = np.ones((k, k))
    structure = structure.astype(bool)

    # opening = erode then dilate
    e = binary_opening(edge_bin, structure=structure)

    # fill small gaps \closing = dilate then erode
    e = binary_closing(e, structure=structure)

    return e