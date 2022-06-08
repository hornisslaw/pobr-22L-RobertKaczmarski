import argparse
import cv2 as cv
import sys

from typing import Optional, Sequence

from color_conversions import BGR2HSV, HSV2BGR, BGR2RGB
from histogram import histogram_equalization
from draw import draw_bounding_box, draw_all_segments
from identification import (
    get_prediction_segments,
    get_prediction_boxes,
    identify_logo_from_segments,
)
from models import SegmentModel
from segmentation import ColorMask, get_segmetns, filter_small_segments
from resize import BicubicInterpolationResizer, NearestNeighbourInterpolationResizer
from filters import (
    close_operation,
    dilation,
    erosion,
    filter_gaussian,
    filter_median,
    filter_ranking,
    open_operation,
)


def main(argv: Optional[Sequence[int]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path")
    parser.add_argument("-r", "--resizer")
    args = parser.parse_args(argv)

    file_path = args.file_path
    resizer_type = args.resizer

    IMAGE_HEIGHT = 400
    IMAGE_WIDTH = 500
    WH_RATIO = 0

    Y_MODEL_INVARIANT_MINIMAS = [0, 0, 0, 0, 0, 0, 0]
    Y_MODEL_INVARIANT_MAXIMAS = [30, 900, 30000, 30000, 75e+7, 817691, 10]
    MODEL_INVARIANT_MINIMAS = [0, 0, 0, 0, 0, 0, 0.42]
    MODEL_INVARIANT_MAXIMAS = [40, 1500, 55000, 55000, 28e+8, 2e+6, 13]
    models = {
        "blue_background_model": SegmentModel(
            WH_RATIO, MODEL_INVARIANT_MINIMAS, MODEL_INVARIANT_MAXIMAS
        ),
        "blue_left_L_model": SegmentModel(
            WH_RATIO, MODEL_INVARIANT_MINIMAS, MODEL_INVARIANT_MAXIMAS
        ),
        "blue_right_L_model": SegmentModel(
            WH_RATIO, MODEL_INVARIANT_MINIMAS, MODEL_INVARIANT_MAXIMAS
        ),
        "blue_D_model": SegmentModel(
            WH_RATIO, MODEL_INVARIANT_MINIMAS, MODEL_INVARIANT_MAXIMAS
        ),
        "yellow_circle_mode": SegmentModel(
            WH_RATIO, Y_MODEL_INVARIANT_MINIMAS, Y_MODEL_INVARIANT_MAXIMAS
        ),
        "red_ring_model": SegmentModel(
            WH_RATIO, MODEL_INVARIANT_MINIMAS, MODEL_INVARIANT_MAXIMAS
        ),
        "red_dot_model": SegmentModel(
            WH_RATIO, MODEL_INVARIANT_MINIMAS, MODEL_INVARIANT_MAXIMAS
        ),
        "red_I_model": SegmentModel(
            WH_RATIO, MODEL_INVARIANT_MINIMAS, MODEL_INVARIANT_MAXIMAS
        ),
    }

    img = cv.imread(file_path)

    if img is None:
        sys.exit("Could not read the image.")
    if resizer_type == "bicubic":
        print(
            f"Resizing image to ({IMAGE_HEIGHT}, {IMAGE_WIDTH}) with bicubic interplation"
        )
        resizer = BicubicInterpolationResizer(img, IMAGE_HEIGHT, IMAGE_WIDTH)
        img = resizer.resize()
    elif resizer_type == "nearest":
        print(
            f"Resizing image to ({IMAGE_HEIGHT}, {IMAGE_WIDTH}) with nearest neighbour interpolation"
        )
        resizer = NearestNeighbourInterpolationResizer(img, IMAGE_HEIGHT, IMAGE_WIDTH)
        img = resizer.resize()
    elif resizer_type:
        sys.exit("ERROR: Wrong name for parameter resizer, exiting...")
    else:
        print("Parsing image withour resize")

    img_rgb = BGR2RGB(img)
    cv.imshow(file_path, img)
    # median = cv.medianBlur(img, 3)
    # img = filter_gaussian(img)
    # cv.imshow("Po gaus", img)
    # img = filter_median(img)
    # cv.imshow("Po median", img)
    # img = filter_ranking(img, 0)
    # cv.imshow("Po ranking", img)

    img_hsv = BGR2HSV(img)
    # cv.imshow("Konwersja BGR do HSV", img_hsv)

    # img_hsv = histogram_equalization(img_hsv, 0)
    # cv.imshow("Po eq", img_equ)

    print("Creating color masks")
    blue = ColorMask(h_min=210, s_min=70, v_min=100, h_max=270, s_max=255, v_max=255)
    red = ColorMask(h_min=300, s_min=50, v_min=50, h_max=360, s_max=255, v_max=255)
    yellow = ColorMask(h_min=20, s_min=70, v_min=100, h_max=60, s_max=255, v_max=255)
    blue_mask = blue.create_mask(img_hsv)
    red_mask = red.create_mask(img_hsv)
    yellow_mask = yellow.create_mask(img_hsv)
    # cv.imshow("Po red", red_mask)
    # cv.imshow("Po yellow", yellow_mask)
    # cv.imshow("Po blue", blue_mask)

    # red_mask = dilation(red_mask, 5)
    # cv.imshow("Po red dil", red_mask)
    # blue_mask = close_operation(blue_mask, 3)
    # cv.imshow("Po blue close", blue_mask)
    # cv.imshow("Yellow po dilation", yellow_mask)

    print("Getting segments")
    yellow_segments, yellow_colored_segments_image = get_segmetns(yellow_mask, "yellow")
    # red_segments, red_colored_segments_image = get_segmetns(red_mask, "red")
    blue_segments, blue_colored_segments_image = get_segmetns(blue_mask, "green")
    # print(f"{len(yellow_segments)=}")
    # print(f"{len(red_segments)=}")
    # print(f"{len(blue_segments)=}")
    # cv.imshow("Red segments", red_colored_segments_image)
    # cv.imshow("Yellow segments", yellow_colored_segments_image)
    # cv.imshow("Blue segments", blue_colored_segments_image)

    print("Filtering small segments")
    yellow_segments_filtered = filter_small_segments(yellow_segments, min_area=50)
    blue_segments_filtered = filter_small_segments(blue_segments, min_area=50)
    # red_segments_filtered = filter_small_segments(red_segments, min_area=50)
    # print(f"{len(yellow_segments_filtered)=}")
    # print(f"{len(red_segments_filtered)=}")
    # print(f"{yellow_segments_filtered[0].wh_ratio=}")

    # draw_all_segments(img_rgb, yellow_segments_filtered)
    # draw_all_segments(img_rgb, blue_segments_filtered)

    print("Identifying")
    yellow_prediction_segments = get_prediction_segments(
        models, yellow_segments_filtered, segment_type="yellow_circle_mode"
    )
    yellow_bounding_boxes = get_prediction_boxes(yellow_prediction_segments)
    # draw_bounding_box(img_rgb, yellow_bounding_boxes)
    # print(f"{len(yellow_prediction_segments)=}")

    blue_prediction_segments = get_prediction_segments(
        models, blue_segments_filtered, segment_type="blue_background_model"
    )
    blue_bounding_boxes = get_prediction_boxes(blue_prediction_segments)

    logo_bounding_boxes = identify_logo_from_segments(
        blue_prediction_segments, yellow_prediction_segments
    )
    draw_bounding_box(img_rgb, logo_bounding_boxes)
    # print(f"{len(yellow_prediction_segments)=}")
    # draw_bounding_box(img_rgb, yellow_bounding_boxes)
    # draw_bounding_box(img_rgb, blue_bounding_boxes)

    # img_bgr = HSV2BGR(img_hsv)
    # cv.imshow("Konwersja HSV do BGR", img_bgr)
    k = cv.waitKey(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
