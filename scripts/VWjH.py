import os
import re
import matplotlib.pyplot as plt

# Directory where your logs are stored
log_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3"
log_files = [f for f in os.listdir(log_dir) if f.startswith("training_log") and f.endswith(".txt")]

# Sort log files to process them in order
log_files.sort()

train_dice = []
val_dice = []

# Parse all log files
for file in log_files:
    with open(os.path.join(log_dir, file), 'r') as f:
        for line in f:
            match = re.search(r'train dice:\s*([0-9.]+)\s*val dice:\s*([0-9.]+)', line)
            if match:
                train_dice.append(float(match.group(1)))
                val_dice.append(float(match.group(2)))

# Create output folder
plot_dir = os.path.join(log_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, "dice_plot.png")

# Plot and save
plt.figure(figsize=(10, 6))
plt.plot(train_dice, label='Train Dice', linewidth=2)
plt.plot(val_dice, label='Val Dice', linewidth=2)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Dice Score", fontsize=14)
plt.title("Train vs Val Dice over Epochs", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f"Dice plot saved to: {plot_path}")
