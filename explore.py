import matplotlib.pyplot as plt
from typing import Optional
from cv2.typing import MatLike


def show_image(
    image: MatLike,
    title: str = "Image",
    cmap: Optional[str] = None,
) -> None:
    """Display an image using matplotlib."""

    # Set the figure size
    FIGX = 8
    FIGY = 6

    plt.figure(figsize=(FIGX, FIGY))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()
