import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


import numpy as np


def expected_calibration_error(confidences, correctness, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.array(confidences)
    correctness = np.array(correctness)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Find items within this confidence bin
        mask = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        if np.any(mask):
            bin_acc = np.mean(correctness[mask])
            bin_conf = np.mean(confidences[mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / len(confidences)

    return ece


# Confidence values (C1–C9 for each test)
control_conf = [0.884, 0.748, 0.824, 0.746, 0.876, 0.892, 0.947, 0.776, 0.887]
baseline_conf = [0.898, 0.861, 0.922, 0.767, 0.882, 0.765, 0.938, 0.711, 0.911]
fewshot_conf = [0.927, 0.825, 0.913, 0.747, 0.831, 0.817, 0.931, 0.934, 0.958]
chain_conf = [0.824, 0.766, 0.74, 0.747, 0.816, 0.805, 0.925, 0.891, 0.966]
combined_conf = [0.801, 0.882, 0.913, 0.881, 0.76, 0.794, 0.937, 0.958, 0.818]

# Correctness (A1–A9 for each test)
control_acc = [1, 0, 0, 1, 1, 0, 1, 0, 1]
baseline_acc = [0, 1, 1, 0, 0, 1, 1, 0, 1]
fewshot_acc = [1, 0, 1, 1, 0, 1, 1, 1, 1]
chain_acc = [1, 0, 1, 1, 0, 1, 1, 1, 1]
combined_acc = [0, 1, 1, 1, 1, 1, 1, 1, 1]

experiments = {
    "Control": (control_conf, control_acc),
    "Baseline": (baseline_conf, baseline_acc),
    "Baseline + Few Shot": (fewshot_conf, fewshot_acc),
    "Baseline + Chain of Thought": (chain_conf, chain_acc),
    "Baseline + Few Shot + Chain": (combined_conf, combined_acc),
}

print("\n=== ROC AUC and ECE Results ===")
for name, (conf, acc) in experiments.items():

    try:
        roc = roc_auc_score(acc, conf)
    except ValueError:
        roc = np.nan  # In case there's not both classes present
    ece = expected_calibration_error(conf, acc)
    print(f"{name:30s} | ROC AUC: {roc:.3f} | ECE: {ece:.3f}")

# --- Temperature experiment data ---
# Columns: Confidence, Accuracy (0/1)
temp_025_conf = [0.957, 0.903, 0.978, 0.964, 0.923, 0.902]
temp_025_acc = [0, 1, 1, 1, 1, 1]

temp_05_conf = [0.912, 0.927, 0.864, 0.814, 0.901, 0.725]
temp_05_acc = [0, 1, 0, 0, 1, 0]

temp_075_conf = [0.731, 0.775, 0.869, 0.716, 0.764, 0.658]
temp_075_acc = [0, 1, 1, 0, 1, 1]

temp_1_conf = [0.64, 0.616, 0.618, 0.592, 0.6394, 0.554]
temp_1_acc = [0, 0, 1, 0, 1, 0]

# Combine into a dictionary
temperature_experiments = {
    "Temp 0.25": (temp_025_conf, temp_025_acc),
    "Temp 0.5": (temp_05_conf, temp_05_acc),
    "Temp 0.75": (temp_075_conf, temp_075_acc),
    "Temp 1": (temp_1_conf, temp_1_acc),
}

# Append these to your existing experiments dictionary
experiments.update(temperature_experiments)

# Now you can call the same functions for ROC AUC / ECE as before:
for name, (conf, acc) in experiments.items():
    try:
        roc = roc_auc_score(acc, conf)
    except ValueError:
        roc = np.nan
    ece = expected_calibration_error(conf, acc)
    print(f"{name:25s} | ROC AUC: {roc:.3f} | ECE: {ece:.3f}")
