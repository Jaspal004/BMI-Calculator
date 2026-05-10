# AI-Based BMI Estimation

A Flask web app that estimates BMI from a full-body image using MediaPipe pose landmarks, silhouette-derived body features, and a Random Forest regression model.

This repository is prepared for public upload without personal photos, names, IDs, or local runtime history.

## Features

- Upload a full-body image and estimate BMI.
- Classify the result as underweight, normal, overweight, or obese.
- Display a reliability score for extracted body features.
- Launch a live camera BMI estimation workflow.
- Track local BMI prediction history and render a trend chart.
- Generate and evaluate a synthetic training dataset.

## Project Structure

```text
app.py                  Flask web app
body_features.py        Feature extraction and reliability scoring
pose_estimation.py      Image-based pose feature extraction
bmi_model.py            Model loading and prediction
bmi_categories.py       BMI category helpers
live_camera.py          Camera-based BMI estimation
generate_dataset.py     Synthetic dataset generator
train_model.py          Model training script
evaluate_model.py       Evaluation and plot generation
templates/index.html    Web UI
project_poster.html     Project poster source
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

To enable Flask debug mode locally:

```powershell
$env:FLASK_DEBUG="1"
python app.py
```

## Training And Evaluation

Regenerate the dataset:

```powershell
python generate_dataset.py
```

Train the model:

```powershell
python train_model.py
```

Evaluate the model:

```powershell
python evaluate_model.py
```

## Privacy Notes

Uploaded images are stored only in `static/uploads/` during local use. That folder is ignored by Git so personal images are not committed. Prediction history is written to `history.csv`, which is also ignored.

`history.example.csv` is included as a blank reference file.

## License

MIT License. See [LICENSE](LICENSE).
