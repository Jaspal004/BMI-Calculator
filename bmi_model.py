import pickle
import sys
import pandas as pd

# Compatibility shim for pickled scikit-learn and numpy objects saved with newer module paths.
# Older sklearn versions used sklearn.ensemble.forest and sklearn.tree.tree.
# Newer sklearn versions may reference sklearn.ensemble._forest, sklearn.tree._classes, or numpy._core.
try:
    import sklearn.ensemble._forest  # noqa: F401
except ModuleNotFoundError:
    try:
        import sklearn.ensemble.forest as _old_forest
        sys.modules["sklearn.ensemble._forest"] = _old_forest
    except ImportError:
        pass

try:
    import sklearn.tree._classes  # noqa: F401
except ModuleNotFoundError:
    try:
        import sklearn.tree.tree as _old_tree
        sys.modules["sklearn.tree._classes"] = _old_tree
    except ImportError:
        pass

try:
    import numpy._core  # noqa: F401
except ModuleNotFoundError:
    try:
        import numpy.core as _old_numpy_core
        sys.modules["numpy._core"] = _old_numpy_core
    except ImportError:
        pass

# ===== FEATURE NAMES (MUST MATCH TRAINING) =====
FEATURE_NAMES = [
    "shoulder","hip","height","torso",
    "arm","leg","ratio",
    "shoulder_h","hip_h"
]


# ===== LOAD MODEL =====
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully!")
    print("Model expects:", getattr(model, "n_features_in_", len(FEATURE_NAMES)))

except Exception as e:
    print("Error loading model:", e)
    model = None


# ===== PREDICT FUNCTION =====
def predict_bmi(features):
    try:
        if model is None:
            print("Model not loaded")
            return None

        # ===== CHECK FEATURE LENGTH =====
        if len(features) != len(FEATURE_NAMES):
            print(f"Feature length mismatch: Expected {len(FEATURE_NAMES)}, got {len(features)}")
            return None

        # ===== CONVERT TO DATAFRAME =====
        df = pd.DataFrame([features], columns=FEATURE_NAMES)

        # ===== PREDICTION =====
        pred = model.predict(df)[0]

        # ===== FINAL CLAMP =====
        pred = max(10, min(45, pred))

        return round(float(pred), 2)

    except Exception as e:
        print("Prediction error inside bmi_model:", e)
        return None
