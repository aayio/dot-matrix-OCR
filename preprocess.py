import cv2
import numpy as np
from cv2.typing import MatLike
from explore import show_image


def resize(img: MatLike) -> MatLike:
    """Resizes the image to a standard width, preserving aspect ratio."""

    # Constants
    STANDARD_WIDTH = 1600

    original_height, original_width = img.shape[:2]
    if original_width == STANDARD_WIDTH:
        return img  # No resize needed

    aspect_ratio = original_height / original_width
    target_height = int(STANDARD_WIDTH * aspect_ratio)

    # Choose interpolation based on whether we are shrinking or enlarging
    if STANDARD_WIDTH < original_width:
        interpolation = cv2.INTER_AREA  # Good for shrinking
    else:
        interpolation = cv2.INTER_LANCZOS4  # Good for enlarging (higher quality)

    resized_img = cv2.resize(
        img, (STANDARD_WIDTH, target_height), interpolation=interpolation
    )
    return resized_img


def preprocess_multi_step(
    original: MatLike, show_intermediates: bool = False
) -> MatLike:
    """Preprocess the image to optimise for character detection."""

    # Set constants
    GRAY_THRESH = 80
    DILATION_KERNEL = np.ones((7, 3), np.uint8)  # (vert, horiz)
    # EROSION_KERNEL = np.ones((1, 5), np.uint8)  # (vert, horiz)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, GRAY_THRESH, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(bw, DILATION_KERNEL, iterations=1)
    # eroded = cv2.erode(dilated, EROSION_KERNEL, iterations=1)

    if show_intermediates:
        show_image(gray, title="Grayscale", cmap="gray")
        show_image(bw, title="Binary Thresholded", cmap="gray")
        show_image(dilated, title="Dilated", cmap="gray")
        # show_image(eroded, title="Eroded", cmap="gray")

    return dilated
