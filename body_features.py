import numpy as np


FEATURE_NAMES = [
    "shoulder", "hip", "height", "torso",
    "arm", "leg", "ratio", "shoulder_h", "hip_h"
]
FEATURE_RANGES = {
    "shoulder": (0.14, 0.58),
    "hip": (0.14, 0.70),
    "height": (0.88, 0.96),
    "torso": (0.20, 0.65),
    "arm": (0.18, 0.55),
    "leg": (0.30, 0.70),
    "ratio": (0.45, 2.20),
    "shoulder_h": (0.15, 0.65),
    "hip_h": (0.15, 0.76),
}

REFERENCE_HEIGHT = 0.92
VISIBILITY_THRESHOLD = 0.5
SEGMENTATION_THRESHOLD = 0.25


def _clamp(value, low, high):
    return min(max(value, low), high)


def _avg_y(landmarks, left_index, right_index):
    return (landmarks[left_index].y + landmarks[right_index].y) / 2


def _range_penalty(value, low, high):
    span = high - low
    if span <= 0:
        return 0.0

    if value < low:
        return min(0.2, ((low - value) / span) * 1.5)
    if value > high:
        return min(0.2, ((value - high) / span) * 1.5)

    edge = span * 0.08
    if edge <= 0:
        return 0.0
    if value < low + edge:
        return ((low + edge - value) / edge) * 0.03
    if value > high - edge:
        return ((value - (high - edge)) / edge) * 0.03

    return 0.0


def estimate_reliability(features):
    if features is None or len(features) != len(FEATURE_NAMES):
        return None

    values = {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, features)
    }

    score = 1.0
    for name, (low, high) in FEATURE_RANGES.items():
        score -= _range_penalty(values[name], low, high)

    expected_ratio = values["shoulder"] / values["hip"]
    if abs(expected_ratio - values["ratio"]) > 0.05:
        score -= 0.12

    if abs((values["shoulder"] / values["height"]) - values["shoulder_h"]) > 0.05:
        score -= 0.08

    if abs((values["hip"] / values["height"]) - values["hip_h"]) > 0.05:
        score -= 0.08

    return round(_clamp(score, 0.1, 0.99), 2)


def _contiguous_width(mask_row, center_col):
    columns = np.flatnonzero(mask_row)
    if len(columns) == 0:
        return None

    breaks = np.where(np.diff(columns) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(columns) - 1]

    best_start = columns[starts[0]]
    best_end = columns[ends[0]]
    best_distance = float("inf")

    for start_index, end_index in zip(starts, ends):
        start = columns[start_index]
        end = columns[end_index]
        if start <= center_col <= end:
            return end - start + 1

        distance = min(abs(center_col - start), abs(center_col - end))
        if distance < best_distance:
            best_distance = distance
            best_start = start
            best_end = end

    return best_end - best_start + 1


def _silhouette_width(segmentation_mask, y_norm, center_x_norm, scale):
    if segmentation_mask is None:
        return None

    mask = segmentation_mask > SEGMENTATION_THRESHOLD
    height_px, width_px = mask.shape
    y = int(_clamp(y_norm, 0, 1) * (height_px - 1))
    center_col = int(_clamp(center_x_norm, 0, 1) * (width_px - 1))
    window = max(1, int(height_px * 0.01))

    rows = mask[max(0, y - window):min(height_px, y + window + 1)]
    row = rows.mean(axis=0) >= 0.35
    width = _contiguous_width(row, center_col)
    if width is None:
        return None

    return (width / width_px) * scale


def _silhouette_band_widths(segmentation_mask, y_values, center_x_norm, scale):
    widths = []
    for y_value in y_values:
        width = _silhouette_width(segmentation_mask, y_value, center_x_norm, scale)
        if width is not None:
            widths.append(width)
    return widths


def _valid_outline_width(width, min_width, max_width):
    if width is None:
        return None
    if width < min_width or width > max_width:
        return None
    return float(width)


def landmarks_to_features(landmarks, segmentation_mask=None, debug=False):
    required_points = [0, 11, 12, 15, 16, 23, 24, 27, 28]
    if any(landmarks[index].visibility < VISIBILITY_THRESHOLD for index in required_points):
        return None

    top_y = landmarks[0].y
    foot_y = max(landmarks[27].y, landmarks[28].y)
    body_height = abs(foot_y - top_y)

    if body_height <= 0:
        return None

    scale = REFERENCE_HEIGHT / body_height

    shoulder_y = _avg_y(landmarks, 11, 12)
    hip_y = _avg_y(landmarks, 23, 24)
    body_center_x = (
        landmarks[11].x + landmarks[12].x +
        landmarks[23].x + landmarks[24].x
    ) / 4
    waist_y = shoulder_y + ((hip_y - shoulder_y) * 0.65)
    abdomen_y_values = [
        shoulder_y + ((hip_y - shoulder_y) * fraction)
        for fraction in [0.40, 0.55, 0.70, 0.85, 1.00]
    ]

    skeleton_shoulder = abs(landmarks[11].x - landmarks[12].x) * scale
    skeleton_hip = abs(landmarks[23].x - landmarks[24].x) * scale

    outline_shoulder = _silhouette_width(
        segmentation_mask, shoulder_y, body_center_x, scale
    )
    outline_waist = _silhouette_width(
        segmentation_mask, waist_y, body_center_x, scale
    )
    outline_hip = _silhouette_width(
        segmentation_mask, hip_y, body_center_x, scale
    )
    outline_abdomen_widths = _silhouette_band_widths(
        segmentation_mask, abdomen_y_values, body_center_x, scale
    )

    # MediaPipe segmentation can sometimes include light backgrounds or arms.
    # Keep outline measurements only when they are plausible next to the pose.
    outline_shoulder = _valid_outline_width(
        outline_shoulder,
        skeleton_shoulder * 0.8,
        min(0.62, max(skeleton_shoulder * 2.0, skeleton_shoulder + 0.12))
    )
    outline_waist = _valid_outline_width(
        outline_waist,
        skeleton_hip * 0.9,
        min(0.72, max(skeleton_shoulder * 2.25, skeleton_hip * 3.0))
    )
    outline_hip = _valid_outline_width(
        outline_hip,
        skeleton_hip * 0.9,
        min(0.72, max(skeleton_shoulder * 2.35, skeleton_hip * 3.2))
    )
    outline_abdomen_widths = [
        _valid_outline_width(
            width,
            skeleton_hip * 0.9,
            min(0.76, max(skeleton_shoulder * 2.35, skeleton_hip * 3.3))
        )
        for width in outline_abdomen_widths
    ]
    outline_abdomen_widths = [
        width for width in outline_abdomen_widths
        if width is not None
    ]

    shoulder = skeleton_shoulder
    if outline_shoulder is not None:
        shoulder = (skeleton_shoulder * 0.8) + (outline_shoulder * 0.2)

    hip_base = max(skeleton_hip * 1.15, skeleton_hip + 0.02)
    outline_body_widths = [
        width for width in [outline_waist, outline_hip]
        if width is not None
    ]
    if outline_abdomen_widths:
        outline_body_widths.append(float(np.median(outline_abdomen_widths)))

    hip = hip_base
    outline_body = None
    outline_was_capped = False
    if outline_body_widths:
        # The widest band often includes relaxed arms, especially in front-view photos.
        outline_body = float(np.median(outline_body_widths))

        skeleton_ratio = skeleton_shoulder / skeleton_hip
        if (
            skeleton_hip < 0.22 and
            skeleton_ratio > 1.65 and
            outline_body < 0.62
        ):
            spillover_cap = max(skeleton_hip * 1.55, skeleton_shoulder * 0.9)
            outline_body = min(outline_body, spillover_cap)
            outline_was_capped = True

        hip = (hip_base * 0.45) + (outline_body * 0.55)

    torso = abs(shoulder_y - hip_y) * scale
    arm = (
        abs(landmarks[11].y - landmarks[15].y) +
        abs(landmarks[12].y - landmarks[16].y)
    ) / 2 * scale
    leg = (
        abs(landmarks[23].y - landmarks[27].y) +
        abs(landmarks[24].y - landmarks[28].y)
    ) / 2 * scale

    shoulder = _clamp(shoulder, 0.14, 0.58)
    hip = _clamp(hip, 0.14, 0.70)
    torso = _clamp(torso, 0.20, 0.65)
    arm = _clamp(arm, 0.18, 0.55)
    leg = _clamp(leg, 0.30, 0.70)
    height = REFERENCE_HEIGHT

    ratio = shoulder / hip
    shoulder_h = shoulder / height
    hip_h = hip / height

    features = [
        shoulder, hip, height, torso,
        arm, leg, ratio, shoulder_h, hip_h
    ]

    if debug:
        print(
            "Feature debug:",
            "skeleton_shoulder=", round(float(skeleton_shoulder), 3),
            "skeleton_hip=", round(float(skeleton_hip), 3),
            "outline_shoulder=", None if outline_shoulder is None else round(float(outline_shoulder), 3),
            "outline_waist=", None if outline_waist is None else round(float(outline_waist), 3),
            "outline_hip=", None if outline_hip is None else round(float(outline_hip), 3),
            "abdomen_widths=", [round(float(width), 3) for width in outline_abdomen_widths],
            "outline_body=", None if outline_body is None else round(float(outline_body), 3),
            "outline_capped=", outline_was_capped,
        )

    return [float(value) for value in features]
