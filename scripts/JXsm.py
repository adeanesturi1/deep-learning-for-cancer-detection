import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns

output_dir = "/sharedscratch/an252/cancerdetectiondataset/output/brats"
os.makedirs(output_dir, exist_ok=True)
metadata_path = os.path.join(output_dir, "preprocessing_metadata_with_splits.csv")
df = pd.read_csv(metadata_path)

assert "PatientID" in df.columns, "Metadata must include 'PatientID'"
assert "Grade" in df.columns, "Metadata must include 'Grade' (HGG or LGG)"

df["Subgroup"] = df["Grade"].map(lambda x: "HGG" if x == "HGG" else "LGG")

dice_scores = []
volume_bins = []
patient_ids = []
nonzero_voxels = []

for _, row in df.iterrows():
    pid = row["PatientID"]
    grade = row["Subgroup"]

    # Synthetic tumour mask
    true_mask = np.zeros((64, 64))
    x, y = np.random.randint(20, 44), np.random.randint(20, 44)
    size = np.random.randint(3, 8)
    true_mask[x-size:x+size, y-size:y+size] = 1

    # Apply Gaussian blur
    sigma = np.random.uniform(1.0, 2.5)
    blurred = gaussian_filter(true_mask, sigma=sigma)
    pred_mask = (blurred > 0.3).astype(np.uint8)

    # Add false positives — more for LGG if you want to simulate lower Dice
    fp_mask = np.zeros_like(true_mask)
    n_fp = 5 if grade == "HGG" else 15
    for _ in range(n_fp):
        xi, yi = np.random.randint(0, 64), np.random.randint(0, 64)
        if true_mask[xi, yi] == 0:
            fp_mask[xi, yi] = 1
    pred_mask = np.clip(pred_mask + fp_mask, 0, 1)

    # Binary true mask
    true_mask = (true_mask > 0).astype(np.uint8)

    # Dice Score
    intersection = np.sum(pred_mask * true_mask)
    dice = (2. * intersection) / (np.sum(pred_mask) + np.sum(true_mask) + 1e-8)

    dice_scores.append(dice)
    voxels = np.sum(true_mask)
    nonzero_voxels.append(voxels)
    volume_bins.append("Low" if voxels < 100 else "High")
    patient_ids.append(pid)

results_df = pd.DataFrame({
    "PatientID": patient_ids,
    "DiceScore": dice_scores,
    "TumourVolume": nonzero_voxels,
    "VolumeBin": volume_bins
})

merged_df = pd.merge(df, results_df, on="PatientID")
csv_path = os.path.join(output_dir, "simulated_dice_scores_with_blur_fp.csv")
merged_df.to_csv(csv_path, index=False)

grade_stats = merged_df.groupby("Grade")["DiceScore"].agg(["mean", "std", "count"])
print("\nGrade-wise Dice Score Statistics:\n", grade_stats)

y_true = (merged_df["Grade"] == "HGG").astype(int)
y_score = merged_df["DiceScore"]

auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else None
if auroc:
    print(f"\nAUROC for HGG vs LGG classification using Dice as proxy: {auroc:.3f}")
else:
    print("\nAUROC not computable: Only one class present.")


# 1. Dice Score Histogram
plt.figure(figsize=(8, 5))
sns.histplot(merged_df["DiceScore"], bins=30, kde=True, color='skyblue')
plt.title("Dice Score Distribution")
plt.xlabel("Dice Score")
plt.ylabel("Patient Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dice_score_distribution.png"))
plt.close()

# 2. Dice Score by Grade
plt.figure(figsize=(6, 5))
sns.boxplot(data=merged_df, x="Grade", y="DiceScore", palette={"HGG": "red", "LGG": "blue"})
plt.title("Dice Score by Tumour Grade")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dice_by_grade.png"))
plt.close()

# 3. Dice Score by Volume Bin
plt.figure(figsize=(6, 5))
sns.boxplot(data=merged_df, x="VolumeBin", y="DiceScore", palette="Set2")
plt.title("Dice Score by Tumour Volume")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dice_by_volume_bin.png"))
plt.close()

# 4. ROC Curve (if applicable)
if auroc:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: HGG vs LGG")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve_hgg_vs_lgg.png"))
    plt.close()

print("Simulation complete. CSV and visualisations saved.")

# === PRINT SUMMARY ===
dice_mean = merged_df["DiceScore"].mean()
dice_std = merged_df["DiceScore"].std()

print("\n===== PERFORMANCE SUMMARY =====")
print(f"Dice Score (Mean ± Std): {dice_mean:.3f} ± {dice_std:.3f}")
if auroc:
    print(f"AUROC (HGG vs LGG): {auroc:.3f}")
else:
    print("AUROC not computable (only one class present)")
print("================================\n")

