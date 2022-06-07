import numpy as np
import math


class BicubicInterpolationResizer:
    def __init__(self, image, new_y, new_x, a: float = -0.75) -> None:
        self.a = a
        self.image = image
        self.new_y = new_y
        self.new_x = new_x

    def resize(self) -> np.ndarray:
        y_ratio = self.image.shape[0] / self.new_y
        x_ratio = self.image.shape[1] / self.new_x
        resized_image = np.zeros((self.new_y, self.new_x, 3), dtype=self.image.dtype)

        for i in range(0, self.new_y):
            for j in range(0, self.new_x):
                y = i * y_ratio
                x = j * x_ratio
                y0 = int(y)
                x0 = int(x)
                dy = y - y0
                dx = x - x0

                y_coeffs = self._evaluate_coefficients(dy)
                x_coeffs = self._evaluate_coefficients(dx)

                intermediaryResults = np.zeros((4, 3), dtype=np.float32)
                for n in range(0, 4):
                    for m in range(0, 4):
                        ppp = self.image[self._check_column(y0, n - 1)][
                            self._check_row(x0, m - 1)
                        ]
                        for index, p in enumerate(ppp):
                            intermediaryResults[n][index] += p * x_coeffs[m]

                result = np.zeros(3)
                for k in range(0, 4):
                    for l in range(0, 3):
                        result[l] += intermediaryResults[k][l] * y_coeffs[k]

                for r in result:
                    r = self._clip(r)

                resized_image[i][j] = result

        return resized_image

    def _check_row(self, x0: int, i: int) -> int:
        if x0 + i > self.image.shape[1] - 1 or x0 + i < 0:
            return x0
        else:
            return x0 + i

    def _check_column(self, y0: int, i: int) -> int:
        if y0 + i > self.image.shape[0] - 1 or y0 + i < 0:
            return y0
        else:
            return y0 + i

    def _evaluate_coefficients(self, x: float) -> list[float]:
        coeffs = np.zeros(4)
        a = self.a
        coeffs[0] = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a
        coeffs[1] = ((a + 2) * x - (a + 3)) * x * x + 1
        coeffs[2] = ((a + 2) * (1 - x) - (a + 3)) * (1 - x) * (1 - x) + 1
        coeffs[3] = 1.0 - coeffs[0] - coeffs[1] - coeffs[2]
        return coeffs

    def _clip(self, value: int):
        if value > 255:
            return 255
        elif value < 0:
            return 0
        else:
            return value


class NearestNeighbourInterpolationResizer:
    def __init__(self, image, new_y, new_x, a: float = -0.75) -> None:
        self.a = a
        self.image = image
        self.new_y = new_y
        self.new_x = new_x

    def resize(self) -> np.ndarray:
        old_height, old_width, _ = self.image.shape
        y_ratio = old_height / self.new_y
        x_ratio = old_width / self.new_x
        resized_image = np.zeros((self.new_y, self.new_x, 3), dtype=self.image.dtype)

        for i in range(0, self.new_y):
            for j in range(0, self.new_x):
                y = i * y_ratio
                x = j * x_ratio
                y0 = int(y)
                x0 = int(x)

                if x0 + 1 >= old_width:
                    x1 = x0
                else:
                    x1 = x0 + 1

                if y0 + 1 >= old_height:
                    y1 = y0
                else:
                    y1 = y0 + 1

                p00 = self._calculate_distance(y0, y0, i, j)
                p10 = self._calculate_distance(y0, x1, i, j)
                p01 = self._calculate_distance(y1, x0, i, j)
                p11 = self._calculate_distance(y1, y1, i, j)

                minimum = np.min([p00, p10, p01, p11])
                pixel = np.array([0, 0, 0], dtype=np.uint8)

                if math.isclose(p00, minimum):
                    pixel = [r for r in self.image[y0][x0]]
                elif math.isclose(p10, minimum):
                    pixel = [r for r in self.image[y1][x1]]
                elif math.isclose(p01, minimum):
                    pixel = [r for r in self.image[y1][x0]]
                elif math.isclose(p11, minimum):
                    pixel = [r for r in self.image[y1][x1]]

                resized_image.itemset((i, j, 0), pixel[0])
                resized_image.itemset((i, j, 1), pixel[1])
                resized_image.itemset((i, j, 2), pixel[2])
        return resized_image

    def _calculate_distance(self, y, x, i, j):
        return np.sqrt(np.power((j - x), 2) + np.power((i - y), 2))
