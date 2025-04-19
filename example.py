import cv2
from explore import show_image
from contours import (
    print_contours_sorted_by_area,
    filter_contours_by_area,
    draw_contours,
)
from preprocess import preprocess_multi_step

# Set constants
MIN_CONTOUR_AREA = 500

image_path = "input.jpg"

image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

show_image(image_rgb, title="Original Image")

preprocessed = preprocess_multi_step(image, show_intermediates=True)

contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total contours: {len(contours)}")

filtered_contours = filter_contours_by_area(contours, MIN_CONTOUR_AREA)

print(f"Filtered contours (area > {MIN_CONTOUR_AREA}): {len(filtered_contours)}")

print_contours_sorted_by_area(filtered_contours)

image_with_contours = draw_contours(base_image=image_rgb, contours=filtered_contours)

show_image(image_with_contours, title="Image with Contours")
