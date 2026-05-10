ESTIMATION_MARGIN = 0.5


def classify_bmi(bmi):
    """Classify pose-based BMI estimates without overreacting near cutoffs."""
    if bmi is None:
        return None

    value = float(bmi)

    if value < 18.5:
        return "Underweight"
    if value < 25.0 + ESTIMATION_MARGIN:
        return "Normal"
    if value < 30.0 + ESTIMATION_MARGIN:
        return "Overweight"
    return "Obese"


def category_color_bgr(category):
    if category == "Underweight":
        return (255, 0, 0)
    if category == "Normal":
        return (0, 255, 0)
    if category == "Overweight":
        return (0, 165, 255)
    return (0, 0, 255)
