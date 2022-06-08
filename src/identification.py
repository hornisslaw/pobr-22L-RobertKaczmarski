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


def yellow_circle_in_blue_backgound(blue_point, yellow_point, error=5):
    blue_y_min, blue_x_min = blue_point[0]
    blue_y_max, blue_x_max = blue_point[1]
    yellow_y_min, yellow_x_min = yellow_point[0]
    yellow_y_max, yellow_x_max = yellow_point[1]
    # print(blue_y_min, blue_x_min, blue_y_max, blue_x_max)
    # print(yellow_y_min, yellow_x_min, yellow_y_max, yellow_x_max)

    if (
        blue_y_min < yellow_y_min + error
        and blue_x_min < yellow_x_min + error
        and blue_y_max > yellow_y_max - error
        and blue_x_max > yellow_x_max - error
    ):
        return True
    else:
        return False


def identify_logo_from_segments(blue_prediction_segments, yellow_prediction_segments):
    blue_bounding_points = []
    yellow_bounding_points = []
    logos_bounding_points = []
    used_yellow = set()

    for b, y in zip(blue_prediction_segments, yellow_prediction_segments):
        blue_bounding_points.append(b.find_bounding_points())
        yellow_bounding_points.append(y.find_bounding_points())

    for blue_point in blue_bounding_points:
        for yellow_point in yellow_bounding_points:
            if yellow_circle_in_blue_backgound(blue_point, yellow_point):
                if yellow_point not in used_yellow:
                    used_yellow.add(yellow_point)
                    logos_bounding_points.append(blue_point)

    return logos_bounding_points
