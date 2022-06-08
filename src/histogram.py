def create_histogram(image):
    histogram = [0 for _ in range(0, 256)]
    for row in image:
        for pixel in row:
            histogram[pixel[0]] += 1
    return histogram


def create_lut(histogram, pixels):
    lut = [0 for _ in range(0, 256)]
    probability_sum = 0
    for i, h in enumerate(histogram):
        probability_sum += h
        lut[i] = probability_sum * 255 / pixels
    return lut


def histogram_equalization(image, pixel_index=0):
    histogram = create_histogram(image)
    pixels = image.size
    look_up_table = create_lut(histogram, pixels)
    new_image = image.copy()

    for i, row in enumerate(new_image):
        for j, pixel in enumerate(row):
            new_image.itemset((i, j, pixel_index), look_up_table[pixel[pixel_index]])

    return new_image
