import joblib
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
import os
import numpy as np

# Load configuration from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get file paths from the configuration
model_path = config['evaluate']['model_path']
X_pred_path = config['evaluate']['X_pred_path']
y_pred_path = config['evaluate']['y_pred_path']
result_path = config['evaluate']['result_path']
metrics_path = config['evaluate']['metrics_path']
roc_plot_path = config['evaluate']['roc_plot_path']

# Ensure the output directories exist
os.makedirs(os.path.dirname(result_path), exist_ok=True)
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
os.makedirs(os.path.dirname(roc_plot_path), exist_ok=True)

# Load the trained model and data
model = joblib.load(model_path)
X = pd.read_csv(X_pred_path)
y = pd.read_csv(y_pred_path)

# Drop the first column if it's just an index
X = X.iloc[:, 1:]
y = y.iloc[:, 1:].values.ravel()

# Predictions
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred, output_dict=True)

# Save full classification report
results = {"accuracy": acc}
for label, metrics in report.items():
    if isinstance(metrics, dict):
        for metric_name, value in metrics.items():
            results[f"{label}_{metric_name}"] = value
    else:
        results[label] = metrics

with open(result_path, "w") as f:
    json.dump(results, f, indent=4)

# Save simplified metrics (for binary compatibility keep '1' class if present)
simple_metrics = {"accuracy": acc}
if '1' in report:
    simple_metrics.update({
        "precision": report['1']['precision'],
        "recall": report['1']['recall']
    })
with open(metrics_path, "w") as f:
    json.dump(simple_metrics, f, indent=4)

# --- Multiclass ROC (One-vs-Rest) ---
if hasattr(model, "predict_proba"):
    y_pred_proba = model.predict_proba(X)
    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)

    roc_data = []
    
    # Generate a more granular set of thresholds
    thresholds = np.linspace(0, 1, 1000)
    
    for i, cls in enumerate(classes):
        # Calculate FPR and TPR for a finer set of thresholds
        fpr_fine, tpr_fine, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])

        # Interpolate the values to a common set of thresholds
        tpr_interpolated = np.interp(thresholds, np.flipud(_), np.flipud(tpr_fine))
        fpr_interpolated = np.interp(thresholds, np.flipud(_), np.flipud(fpr_fine))

        # Add start and end points
        fpr_interpolated = np.insert(fpr_interpolated, 0, 0)
        tpr_interpolated = np.insert(tpr_interpolated, 0, 0)

        # Store the interpolated data points
        for f, t in zip(fpr_interpolated, tpr_interpolated):
            roc_data.append({
                "class": int(cls),
                "fpr": round(float(f), 4),
                "tpr": round(float(t), 4)
            })
    
    with open(roc_plot_path, "w") as f:
        json.dump(roc_data, f, indent=4)