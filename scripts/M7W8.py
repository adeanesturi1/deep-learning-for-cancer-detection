import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Paths ===
dice_path = "/home/an252/deep-learning-for-cancer-detection/bcbm_fold3_val_dice.csv"
output_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/figures"
os.makedirs(output_dir, exist_ok=True)

# === Load Dice Scores ===
df_dice = pd.read_csv(dice_path)

# === Histogram of Dice Scores ===
plt.figure(figsize=(10, 6))
sns.histplot(df_dice["Dice"], bins=20, kde=True)
plt.title("Histogram of Dice Scores (BCBM Fold 3)")
plt.xlabel("Dice Score")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dice_histogram.png"))
plt.close()

# === Boxplot of Dice Scores ===
plt.figure(figsize=(6, 4))
sns.boxplot(y=df_dice["Dice"])
plt.title("Boxplot of Dice Scores")
plt.ylabel("Dice Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dice_boxplot.png"))
plt.close()

# === Optional HER2 Stratification ===
her2_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/bcbm_metadata.csv"
df_meta = pd.read_csv(her2_path)
df_merged = pd.merge(df_dice, df_meta, left_on="Patient", right_on="nnUNet_ID")
df_merged["HER2_Status"] = df_merged["HER2_Status"].fillna("NA")

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_merged, x="HER2_Status", y="Dice")
plt.title("Dice Scores by HER2 Status")
plt.xlabel("HER2 Status")
plt.ylabel("Dice Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dice_by_her2.png"))
plt.close()

print(f"Figures saved to: {output_dir}")
