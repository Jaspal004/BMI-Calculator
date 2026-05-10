from flask import Flask, render_template, request
import os
import subprocess
import csv
from werkzeug.utils import secure_filename
from pose_estimation import extract_features
from bmi_model import predict_bmi
from bmi_categories import classify_bmi
from body_features import estimate_reliability

app = Flask(__name__)

# ===== CONFIG =====
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
EXPECTED_FEATURES = 9
DEBUG_FEATURES = os.environ.get("BMI_DEBUG_FEATURES") == "1"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ===== FILE CHECK =====
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ===== CREATE HISTORY FILE =====
if not os.path.exists("history.csv"):
    with open("history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["BMI", "Category"])


# ===== LOAD HISTORY =====
def load_history(limit=50):
    history = []
    try:
        with open("history.csv", "r") as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) != 2 or row[0] == "BMI":
                    continue

                history.append({
                    "bmi": row[0],
                    "category": row[1]
                })

        history = history[-limit:][::-1]

    except Exception as e:
        print("History error:", e)

    return history


# ===== LOAD GRAPH DATA =====
def load_chart_data(limit=50):
    labels = []
    values = []

    try:
        with open("history.csv", "r") as f:
            reader = csv.reader(f)

            clean = []
            for row in reader:
                if len(row) != 2 or row[0] == "BMI":
                    continue
                clean.append(row)

            clean = clean[-limit:]

            for i, row in enumerate(clean):
                labels.append(str(i + 1))
                try:
                    values.append(float(row[0]))
                except:
                    values.append(0)

    except Exception as e:
        print("Chart error:", e)

    return labels, values


# ===== MAIN ROUTE =====
@app.route("/", methods=["GET", "POST"])
def index():
    bmi = None
    category = None
    reliability = None
    error = None
    image_path = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", error="No file selected")

        if not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type")

        try:
            # ===== SAVE IMAGE =====
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            image_path = path

            # ===== EXTRACT FEATURES =====
            features = extract_features(path, debug=DEBUG_FEATURES)

            if DEBUG_FEATURES:
                print("Feature length:", len(features) if features else "None")

            # ===== VALIDATION =====
            if features is None:
                error = "Pose not detected. Use full-body image."

            elif len(features) != EXPECTED_FEATURES:
                error = f"Feature mismatch: Expected {EXPECTED_FEATURES}, got {len(features)}"

            else:
                bmi = predict_bmi(features)

                if bmi is None:
                    error = "Model prediction failed."

                else:
                    # ===== CLAMP =====
                    bmi = max(15, min(40, bmi))

                    # ===== CATEGORY =====
                    category = classify_bmi(bmi)

                    # ===== RELIABILITY =====
                    reliability = estimate_reliability(features)

                    # ===== SAVE HISTORY =====
                    with open("history.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([round(bmi, 2), category])

        except Exception as e:
            print("Processing error:", e)
            error = "Could not process image. Try another image."

    # ===== LOAD DATA =====
    history = load_history()
    chart_labels, chart_values = load_chart_data()

    return render_template(
        "index.html",
        bmi=bmi,
        category=category,
        reliability=reliability,
        error=error,
        image_path=image_path,
        history=history,
        chart_labels=chart_labels,
        chart_values=chart_values
    )


# ===== CAMERA ROUTE =====
@app.route("/camera")
def camera():
    try:
        subprocess.Popen(["python", "live_camera.py"])
        return """
        <h2>Camera Started</h2>
        <p>Check the popup window.</p>
        <a href="/">Go Back</a>
        """
    except Exception as e:
        return f"Error starting camera: {str(e)}"


# ===== CLEAR HISTORY =====
@app.route("/clear")
def clear():
    with open("history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["BMI", "Category"])

    return """
    <h2>History Cleared</h2>
    <a href="/">Go Back</a>
    """


# ===== RUN =====
if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
