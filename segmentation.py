import numpy as np


class ColorMask:
    def __init__(
        self, h_min: int, s_min: int, v_min: int, h_max: int, s_max: int, v_max: int
    ) -> None:
        self.h_min = h_min
        self.s_min = s_min
        self.v_min = v_min
        self.h_max = h_max
        self.s_max = s_max
        self.v_max = v_max

    def create_mask(self, image: np.ndarray) -> np.ndarray:
        new_image = np.zeros((image.shape[0], image.shape[1]), image.dtype)

        for i, row in enumerate(image):
            for j, pixel in enumerate(row):
                if (
                    (self.h_min <= pixel[0] * 2 <= self.h_max)
                    and (self.s_min <= pixel[1] <= self.s_max)
                    and (self.v_min <= pixel[2] <= self.v_max)
                ):
                    # new_image[i][j] = np.array([255, 255, 255], dtype=np.uint8)
                    new_image[i][j] = 255
                else:
                    # new_image[i][j] = np.array([0, 0, 0], dtype=np.uint8)
                    new_image[i][j] = 0
        return new_image


class Segment:
    def __init__(self, color, points) -> None:
        self.color = color
        self.points = points
        self.area = self._calculate_area()
        self.m00 = self._find_moment(0, 0)
        self.m10 = self._find_moment(1, 0)
        self.m01 = self._find_moment(0, 0)
        self.i_center = self.m10 / self.m00
        self.j_center = self.m01 / self.m00
        self.bounding_box_points = self.find_bounding_points()
        self.wh_ratio = self._calculate_wh_ratio()
        self.invariant_moments = None

    def _calculate_area(self):
        return len(self.points)

    def _find_moment(self, p, q):
        moment = 0.0
        for point in self.points:
            y, x = point
            moment += np.power(x, p) * np.power(y, q)
        return moment

    def _find_central_moment(self, p, q):
        central_moment = 0.0
        for point in self.points:
            y, x = point
            central_moment += np.power(x - self.i_center, p) * np.power(
                y - self.j_center, q
            )
        return central_moment

    def find_bounding_points(self):
        divided_list = list(zip(*self.points))
        y_list = divided_list[0]
        x_list = divided_list[1]
        x_min = np.min(x_list)
        x_max = np.max(x_list)
        y_min = np.min(y_list)
        y_max = np.max(y_list)
        return ((y_min, x_min), (y_max, x_max))

    def find_invariant_moments(self):
        if not self.invariant_moments:
            self.invariant_moments = self._calculate_moments()
        return self.invariant_moments

    def _calculate_wh_ratio(self):
        box_height, box_width = self._calculate_width_and_height(
            self.bounding_box_points
        )
        return box_width / box_height if box_height != 0 else box_width

    def _calculate_width_and_height(self, bounding_box_points):
        y_min, x_min = bounding_box_points[0]
        y_max, x_max = bounding_box_points[1]
        box_width = x_max - x_min
        box_height = y_max - y_min
        return box_height, box_width

    def _calculate_moments(self):
        mc11 = self._find_central_moment(1, 1)
        mc02 = self._find_central_moment(0, 2)
        mc20 = self._find_central_moment(2, 0)
        mc12 = self._find_central_moment(1, 2)
        mc21 = self._find_central_moment(2, 1)
        mc30 = self._find_central_moment(3, 0)
        mc03 = self._find_central_moment(0, 3)
        M1 = (mc20 + mc02) / np.power(self.m00, 2)
        M2 = (np.power((mc20 - mc02), 2) + 4 * np.power(mc11, 2)) / np.power(
            self.m00, 4
        )
        M3 = (
            np.power((mc30 - 3 * mc12), 2) + np.power((3 * mc21 - mc03), 2)
        ) / np.power(self.m00, 5)
        M4 = (np.power((mc30 + mc12), 2) + np.power((mc21 + mc03), 2)) / np.power(
            self.m00, 5
        )
        M5 = (
            (mc30 - 3 * mc12)
            * (mc30 + mc12)
            * (np.power((mc30 + mc12), 2) - 3 * np.power((mc21 + mc03), 2))
            + (3 * mc21 - mc03)
            * (mc21 + mc03)
            * (3 * np.power((mc30 + mc12), 2) - np.power((mc21 + mc03), 2))
        ) / np.power(self.m00, 10)
        M6 = (
            (mc20 - mc02) * ((np.power((mc30 + mc12), 2) - np.power((mc21 + mc03), 2)))
            + 4 * mc11 * (mc30 + mc12) * (mc21 + mc03)
        ) / np.power(self.m00, 7)
        M7 = (mc20 * mc02 - np.power(mc11, 2)) / np.power(self.m00, 4)
        # W3 =
        # W7 =
        return (M1, M2, M3, M4, M5, M6, M7)


def floodfill(row: int, col: int, color, binaryImage, color_name):
    width, height, _ = binaryImage.shape
    points_queue = [(row, col)]
    states = []
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while len(points_queue) > 0:
        point = points_queue.pop(0)

        binaryImage[point] = color
        states.append(point)

        for d in directions:
            new_position = (point[0] + d[0], point[1] + d[1])
            if not is_valid_column(width, height, new_position):
                continue
            if (
                new_position[0] >= 0
                and new_position[0] < height
                and new_position[1] >= 0
                and new_position[1] < width
                and binaryImage[new_position][0] == 255
            ):
                if new_position not in points_queue:
                    points_queue.append(new_position)

    return Segment(color_name, states), binaryImage


def is_valid_column(width, height, new_position):
    y, x = new_position
    if y >= 0 and y < width and x >= 0 and x < height:
        return True
    else:
        return False


def random_color_pixel():
    color = list(np.random.choice(range(254), size=3))
    return color


def get_segmetns(binaryImage: np.ndarray, color_name: str):
    new_binary = np.array(
        [[[p, p, p] for p in row] for row in binaryImage], dtype=np.uint8
    )
    segments = []
    for i in range(new_binary.shape[0]):
        for j in range(new_binary.shape[1]):
            if new_binary[i][j][0] == 255:
                random_color = random_color_pixel()
                segment, colored_image = floodfill(
                    i, j, random_color, new_binary, color_name
                )
                segments.append(segment)
    return segments, colored_image


def filter_small_segments(segments, min_area):
    return list(filter(lambda x: x.area >= min_area, segments))


# def get_prediction_boxes(models, segments, segment_type):
#     prediction_bounding_boxes = []
#     prediction_segments = []

#     for s in segments:
#         if valid_segment(models, s, segment_type):
#             prediction_bounding_boxes.append(s.find_bounding_points())
#             prediction_segments.append(s)

#     return prediction_bounding_boxes


def get_prediction_segments(models, segments, segment_type):
    prediction_segments = []

    for s in segments:
        if valid_segment(models, s, segment_type):
            prediction_segments.append(s)

    return prediction_segments


def get_prediction_boxes(prediction_segments):
    prediction_bounding_boxes = []
    for ps in prediction_segments:
        prediction_bounding_boxes.append(ps.find_bounding_points())
    return prediction_bounding_boxes


def valid_segment(models, segment, segment_type):
    segment_moments = segment.find_invariant_moments()
    return all(
        [
            models[segment_type].check_invariant_moment(i, m)
            for i, m in enumerate(segment_moments)
        ]
    )
