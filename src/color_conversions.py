import numpy as np


def BGR2HSV(image: np.ndarray) -> np.ndarray:
    new_image = image.copy()

    for i, row in enumerate(image):
        for j, pixel in enumerate(row):
            b, g, r = pixel / 255
            minimum = min(b, g, r)
            maximum = max(b, g, r)
            delta = maximum - minimum
            V = maximum
            S = 0 if maximum == 0 else delta / maximum
            if delta == 0:
                H = 0
            elif maximum == b:
                H = 60 * (((r - g) / delta) + 4)
            elif maximum == g:
                H = 60 * (((b - r) / delta) + 2)
            elif maximum == r:
                H = 60 * (((g - b) / delta) % 6)

            new_image.itemset((i, j, 0), H / 2)
            new_image.itemset((i, j, 1), S * 255)
            new_image.itemset((i, j, 2), V * 255)

            # new_image[i][j] = np.array([H / 2, S * 255, V * 255], dtype=np.uint8)

    return new_image

def HSV2BGR(image: np.ndarray) -> np.ndarray:
    new_image = image.copy()

    for i, row in enumerate(image):
        for j, pixel in enumerate(row):
            h, s, v = pixel
            h, s, v = 2 * h, s / 255, v / 255
            c = v * s
            x = c * (1 - np.abs((h / 60) % 2 - 1))
            m = v - c
            if 0 <= h < 60:
                rp, gp, bp = (c, x, 0)
            elif 60 <= h < 120:
                rp, gp, bp = (x, c, 0)
            elif 120 <= h < 180:
                rp, gp, bp = (0, c, x)
            elif 180 <= h < 240:
                rp, gp, bp = (0, x, c)
            elif 240 <= h < 300:
                rp, gp, bp = (x, 0, c)
            elif 300 <= h < 360:
                rp, gp, bp = (c, 0, x)
            
            new_image.itemset((i, j, 0), (bp + m) * 255)
            new_image.itemset((i, j, 1), (gp + m) * 255)
            new_image.itemset((i, j, 2), (rp + m) * 255)

            # new_image[i][j] = np.array(
            #     [(bp + m) * 255, (gp + m) * 255, (rp + m) * 255], dtype=np.uint8
            # )

    return new_image

def BGR2RGB(image: np.ndarray) -> np.ndarray:
    new_image = image.copy()

    for i, row in enumerate(image):
        for j, pixel in enumerate(row):
            b, _, r = pixel
            new_image.itemset((i, j, 0), r)
            new_image.itemset((i, j, 2), b)
            
    return new_image