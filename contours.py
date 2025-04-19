import cv2
from cv2.typing import MatLike
from typing import Sequence


def print_contours_sorted_by_area(contours: Sequence[MatLike]) -> None:
    if not contours:
        print("No contours provided.")
        return

    # Create a list of (area, index) tuples
    contour_info = [(cv2.contourArea(c), i) for i, c in enumerate(contours)]

    # Sort the list based on area (the first element of the tuple)
    contour_info_sorted = sorted(contour_info, key=lambda item: item[0])

    print("Contours sorted by area (ascending):")
    print("-" * 40)
    print(f"{'Index':<10} {'Area':<10}")
    print("-" * 40)
    for area, index in contour_info_sorted:
        print(f"{index:<10} {area:<10.2f}")  # Format area to 2 decimal places
    print("-" * 40)


def filter_contours_by_area(
    contours: Sequence[MatLike], min_area: int
) -> Sequence[MatLike]:
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def draw_contours(
    base_image: MatLike,
    contours: Sequence[MatLike],
) -> MatLike:
    output_image = base_image.copy()

    COLOR_BOX = (255, 0, 0)  # Red
    COLOR_OUTLINE = (0, 255, 0)  # Green
    BOX_LABEL = (255, 0, 0)  # Red
    BOX_THICKNESS = 2
    FONT_SCALE = 1.0

    for i, c in enumerate(contours):
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(output_image, (x, y), (x + w, y + h), COLOR_BOX, BOX_THICKNESS)

        cv2.drawContours(output_image, [c], -1, COLOR_OUTLINE, BOX_THICKNESS)

        # Calculate centroid to place text
        M = cv2.moments(c)
        cx, cy = x, y  # Default to top-left of bbox
        if M["m00"] != 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        # Add index text
        cv2.putText(
            output_image,
            str(i),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            BOX_LABEL,
            BOX_THICKNESS,
        )

    return output_image
