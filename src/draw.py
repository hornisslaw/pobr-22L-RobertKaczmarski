import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def draw_bounding_box(image: np.ndarray, bounding_boxes) -> None:
    fig, ax = plt.subplots()
    ax.imshow(image)
    for b in bounding_boxes:
        y_min, x_min = b[0]
        y_max, x_max = b[1]
        box_width = x_max - x_min
        box_height = y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.show()


def draw_all_segments(img_rgb, segments):
    for s in segments:
        print(f"{s}, ", s.find_invariant_moments())
        print(s.find_bounding_points())
        draw_bounding_box(img_rgb, [s.find_bounding_points()])
