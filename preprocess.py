import cv2
import numpy as np
from cv2.typing import MatLike
from explore import show_image


def preprocess_multi_step(
    original: MatLike, show_intermediates: bool = False
) -> MatLike:
    """Preprocess the image to optimise for character detection."""

    # Set constants
    GRAY_THRESH = 80
    DILATION_KERNEL = np.ones((5, 3), np.uint8)  # (vert, horiz)
    EROSION_KERNEL = np.ones((1, 3), np.uint8)  # (vert, horiz)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, GRAY_THRESH, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(bw, DILATION_KERNEL, iterations=1)
    eroded = cv2.erode(dilated, EROSION_KERNEL, iterations=1)

    if show_intermediates:
        show_image(gray, title="Grayscale", cmap="gray")
        show_image(bw, title="Binary Thresholded", cmap="gray")
        show_image(dilated, title="Dilated", cmap="gray")
        show_image(eroded, title="Eroded", cmap="gray")

    return eroded
