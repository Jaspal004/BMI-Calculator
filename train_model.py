import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ===== LOAD DATA =====
try:
    data = pd.read_csv("dataset.csv")
    print("Dataset loaded successfully!")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# ===== FEATURE SET (9 FEATURES ONLY) =====
features = [
    "shoulder","hip","height","torso",
    "arm","leg","ratio",
    "shoulder_h","hip_h"
]

# ===== VALIDATE COLUMNS =====
for col in features + ["bmi"]:
    if col not in data.columns:
        print(f"Missing column: {col}")
        exit()

# ===== PREPARE DATA =====
X = data[features]
y = data["bmi"]

print("\nTraining features:", list(X.columns))
print("Total samples:", len(X))

# ===== TRAIN / TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== MODEL =====
model = RandomForestRegressor(
    n_estimators=250,
    max_depth=12,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# ===== EVALUATION =====
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== PERFORMANCE =====")
print(f"MAE: {round(mae, 2)}")
print(f"R2: {round(r2, 2)}")

# ===== SAMPLE CHECK =====
sample = X_test.iloc[0]
pred = model.predict(pd.DataFrame([sample], columns=features))[0]

print("\nSample Prediction:")
print("Actual BMI:", round(y_test.iloc[0], 2))
print("Predicted BMI:", round(pred, 2))

# ===== FEATURE IMPORTANCE =====
try:
    importances = model.feature_importances_

    plt.figure()
    plt.barh(features, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance")

    plt.savefig("feature_importance.png")
    print("Saved: feature_importance.png")

    plt.show()
except Exception as e:
    print("Feature importance error:", e)

# ===== SAVE MODEL =====
try:
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\nModel saved successfully!")
    print("Model expects:", getattr(model, "n_features_in_", len(features)))
except Exception as e:
    print("Error saving model:", e)
