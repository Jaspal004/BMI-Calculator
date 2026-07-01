import csv
import random

from body_features import FEATURE_NAMES, REFERENCE_HEIGHT


COLUMNS = FEATURE_NAMES + ["bmi"]


def _clamp(value, low, high):
    return min(max(value, low), high)


def generate_sample(bmi):
    # The app normalizes every detected person to this body-height scale.
    height = REFERENCE_HEIGHT + random.uniform(-0.015, 0.015)

    fat = _clamp((bmi - 15.0) / 25.0, 0, 1)

    if bmi < 20:
        body_types = ["average", "slender", "muscular"]
        weights = [55, 35, 10]
    elif bmi < 25:
        body_types = ["average", "slender", "muscular", "central"]
        weights = [55, 15, 20, 10]
    elif bmi < 30:
        body_types = ["average", "central", "pear", "muscular"]
        weights = [45, 25, 15, 15]
    else:
        body_types = ["average", "central", "pear", "muscular"]
        weights = [30, 45, 15, 10]

    body_type = random.choices(body_types, weights=weights, k=1)[0]

    # These widths represent an outer body silhouette, not skeleton landmarks.
    shoulder_h = random.gauss(0.23 + (0.16 * fat), 0.018)
    hip_h = random.gauss(0.18 + (0.36 * fat), 0.026)
    torso_h = random.gauss(0.30 + (0.16 * fat), 0.026)
    arm_h = random.gauss(0.26 + (0.13 * fat), 0.022)
    leg_h = random.gauss(0.49 + (0.04 * fat), 0.022)

    if body_type == "central":
        hip_h += random.uniform(0.03, 0.08)
        torso_h += random.uniform(0.02, 0.05)
    elif body_type == "pear":
        hip_h += random.uniform(0.04, 0.09)
        shoulder_h -= random.uniform(0.00, 0.025)
    elif body_type == "muscular":
        shoulder_h += random.uniform(0.04, 0.08)
        hip_h -= random.uniform(0.00, 0.03)
        arm_h += random.uniform(0.02, 0.06)
    elif body_type == "slender":
        shoulder_h -= random.uniform(0.015, 0.04)
        hip_h -= random.uniform(0.015, 0.04)

    # Add camera/pose/clothing noise so the model does not learn a perfect formula.
    shoulder_h += random.uniform(-0.018, 0.018)
    hip_h += random.uniform(-0.025, 0.025)
    torso_h += random.uniform(-0.022, 0.022)
    arm_h += random.uniform(-0.018, 0.018)
    leg_h += random.uniform(-0.018, 0.018)

    shoulder_h = _clamp(shoulder_h, 0.16, 0.58)
    hip_h = _clamp(hip_h, 0.16, 0.72)
    torso_h = _clamp(torso_h, 0.22, 0.66)
    arm_h = _clamp(arm_h, 0.18, 0.56)
    leg_h = _clamp(leg_h, 0.34, 0.72)

    shoulder = shoulder_h * height
    hip = hip_h * height
    torso = torso_h * height
    arm = arm_h * height
    leg = leg_h * height

    # ===== DERIVED FEATURES =====
    ratio = shoulder / hip
    shoulder_h = shoulder / height
    hip_h = hip / height

    return [
        round(shoulder, 2),
        round(hip, 2),
        round(height, 2),
        round(torso, 2),
        round(arm, 2),
        round(leg, 2),
        round(ratio, 2),
        round(shoulder_h, 2),
        round(hip_h, 2),
        round(bmi, 2)
    ]


def generate_dataset(path="dataset.csv", samples_per_category=400):
    bmi_ranges = [
        (15.0, 18.4),
        (18.5, 24.9),
        (25.0, 29.9),
        (30.0, 40.0),
    ]

    rows = []
    for low, high in bmi_ranges:
        for _ in range(samples_per_category):
            sample = None
            while sample is None:
                sample = generate_sample(random.uniform(low, high))
            rows.append(sample)

    random.shuffle(rows)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows in {path}")


if __name__ == "__main__":
    generate_dataset()
