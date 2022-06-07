import numpy as np


def erosion(image, filter_size):
    new_image = np.zeros(image.shape, image.dtype)
    offset = int(filter_size/2)
    half_filter = int((filter_size - 1) / 2)

    for i in range(offset, new_image.shape[1] - offset):
        for j in range(offset, new_image.shape[0] - offset):
            minimum = 256
            for n in range(0, filter_size):
                for m in range(0, filter_size):
                    a = i - half_filter - 1 + n
                    b = j - half_filter - 1 + m
                    if image[b][a] < minimum:
                        minimum = image[b][a]
            new_image[j][i] = minimum
    return new_image


def dilation(image, filter_size):
    new_image = np.zeros(image.shape, image.dtype)
    offset = int(filter_size / 2)
    half_filter = int((filter_size - 1) / 2)
    for i in range(offset, new_image.shape[1] - offset):
        for j in range(offset, new_image.shape[0] - offset):
            maximum = 0
            for n in range(0, filter_size):
                for m in range(0, filter_size):
                    a = i - half_filter - 1 + n
                    b = j - half_filter - 1 + m
                    if image[b][a] > maximum:
                        maximum = image[b][a]
            new_image[j][i] = maximum
    return new_image


def open_operation(image, filter_size):
    image_ero = erosion(image, filter_size)
    return dilation(image_ero, filter_size)


def close_operation(image, filter_size):
    image_dil = dilation(image, filter_size)
    return erosion(image_dil, filter_size)


def filter_gaussian(image):
    gaussian_matrix = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])
    img = image.copy()
    rows, cols, channels = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            img[i, j] = convolve(gaussian_matrix, image[i - 1:i + 2, j - 1:j + 2].astype(int))
    return img


def filter_median(image):
    img = image.copy()
    rows, cols, channels = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            img[i, j] = median(image[i - 1:i + 2, j - 1:j + 2].astype(int))
    return img


def filter_ranking(image, rank):
    img = image.copy()
    rows, cols, channels = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            img[i, j] = ranking(image[i - 1:i + 2, j - 1:j + 2].astype(int), rank)
    return img


def median(pixels):
    pixels_list = []
    for i in range(0, 3):
        for j in range(0, 3):
            pixels_list.append([i, j, (int(pixels[i, j, 0]) + int(pixels[i, j, 1]) + int(pixels[i, j, 2])) / 3])
    pixels_list.sort(key=lambda x: x[2])
    return pixels[pixels_list[5][0], pixels_list[5][1]]


def ranking(pixels, rank):
    pixels_list = []
    for i in range(0, 3):
        for j in range(0, 3):
            pixels_list.append([i, j, (int(pixels[i, j, 0]) + int(pixels[i, j, 1]) + int(pixels[i, j, 2])) / 3])
    pixels_list.sort(key=lambda x: x[2])
    return pixels[pixels_list[rank][0], pixels_list[rank][1]]


def convolve(filtr, pixels):
    res = np.zeros(3)
    for i in range(0, 3):
        for j in range(0, 3):
            res = np.add(res, pixels[i, j] * filtr[i, j])
    return np.divide(res, np.full(3, 16))