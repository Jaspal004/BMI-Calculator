import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ===== LOAD DATA =====
data = pd.read_csv("dataset.csv")

# ===== FEATURES =====
features = ["shoulder","hip","height","torso","arm","leg","ratio","shoulder_h","hip_h"]
X = data[features]
y = data["bmi"]

# ===== TRAIN/TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== LOAD MODEL =====
model = pickle.load(open("model.pkl","rb"))

# ===== PREDICT =====
y_pred = model.predict(X_test)

# ===== METRICS =====
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"MAE (Error): {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# ===== SCATTER PLOT =====
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.title("Actual vs Predicted BMI")
plt.grid(True)

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')

# Save plot
plt.savefig("evaluation_plot.png")
print("Saved: evaluation_plot.png")

plt.show()

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
    print("Feature importance not available for this model:", e)
