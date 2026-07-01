# AI-Based BMI Estimation

A Flask web app that estimates BMI from a full-body image using MediaPipe pose landmarks, silhouette-derived body features, and a Random Forest regression model.

This repository is prepared for public upload without personal photos, names, IDs, or local runtime history.
Live Demo :https://bmi-calculator-corp.onrender.com/

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

The main page supports image upload and browser camera capture. Browser camera access works on `localhost` and on HTTPS deployment URLs.

To enable Flask debug mode locally:

```powershell
$env:FLASK_DEBUG="1"
python app.py
```

To show the app on another device on the same Wi-Fi network:

```powershell
$env:FLASK_RUN_HOST="0.0.0.0"
python app.py
```

Then open `http://YOUR_LOCAL_IP:5000` from the other device.

## Live Demo For Job Applications

This is a Flask app, so use a Python web host for the working demo. GitHub Pages is only suitable for static files such as `project_poster.html`.

Recommended Render settings:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
Python Version: 3.11.9
```

Deployment flow:

1. Push this repository to GitHub.
2. Create a Render Web Service from the GitHub repository.
3. Use the build and start commands above.
4. After deployment, add the `onrender.com` URL to your resume, LinkedIn, and GitHub README.
5. Demo with non-private sample images. Do not upload personal photos in public interviews.

Notes:

- The repo includes `.python-version` so Render uses Python 3.11.9.
- The hosted demo uses the browser camera button on the main page.
- The old desktop OpenCV camera route is disabled by default. For local desktop use, set `ENABLE_LOCAL_CAMERA=1` and visit `/camera`.

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
