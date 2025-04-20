import cv2
from explore import show_image
from contours import (
    print_contours_sorted_by_area,
    filter_contours_by_area,
    draw_contours,
)
from preprocess import preprocess_multi_step, resize
from cropPhoto import crop_photo_to_display, crop_edges

# Set constants

# For a normalised width of 1600, the decimal place has an area of approximately 900
MIN_CONTOUR_AREA = 600

image_path = "input.jpg"

uncropped_BGR = cv2.imread(image_path)

uncropped_RGB = cv2.cvtColor(uncropped_BGR, cv2.COLOR_BGR2RGB)

show_image(uncropped_RGB, title="Uncropped Photo")

cropped = crop_photo_to_display(uncropped_BGR)

cropped_RGB = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

show_image(cropped_RGB, title="Cropped Photo")

resized = resize(cropped)

edges_cropped = crop_edges(resized)

edges_cropped_RGB = cv2.cvtColor(
    edges_cropped, cv2.COLOR_BGR2RGB
)  # contours will be drawn on this

preprocessed = preprocess_multi_step(edges_cropped, show_intermediates=True)

contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total contours: {len(contours)}")

filtered_contours = filter_contours_by_area(contours, MIN_CONTOUR_AREA)

print(f"Filtered contours (area > {MIN_CONTOUR_AREA}): {len(filtered_contours)}")

print_contours_sorted_by_area(filtered_contours)

image_with_contours = draw_contours(
    base_image=edges_cropped_RGB, contours=filtered_contours
)

show_image(image_with_contours, title="Image with Contours")
