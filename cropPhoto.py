import cv2
import numpy as np


from cv2.typing import MatLike


def crop_photo_to_display(image: MatLike) -> MatLike:
    """Finds the largest display area in an image and crops it.
    Returns the cropped image, or the original image if cropping fails.
    """

    # Define the BGR color range for the display
    LOWER = np.array([215, 25, 0])
    UPPER = np.array([245, 65, 20])

    # Create a mask
    mask = cv2.inRange(image, LOWER, UPPER)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("No display region found")
        return image

    # Find the largest contour (assumed to be the display)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the original image using the calculated rectangle
    cropped = image[y : y + h, x : x + w]

    return cropped


def crop_edges(image: MatLike) -> MatLike:
    CROP_H_FACTOR = 0.90
    CROP_W_FACTOR = 0.95

    h, w = image.shape[:2]

    new_h = int(h * CROP_H_FACTOR)
    new_w = int(w * CROP_W_FACTOR)

    # Calculate the starting points for the crop (centered)
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2

    # Perform the crop
    cropped_image = image[start_y : start_y + new_h, start_x : start_x + new_w]

    return cropped_image
