import os
import re
import matplotlib.pyplot as plt
import numpy as np

log_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3"
log_files = [f for f in os.listdir(log_dir) if f.startswith("training_log") and f.endswith(".txt")]
log_files.sort()

train_dice = []
val_dice = []
epoch_durations = []

if not log_files:
    print(f"Error: No training log files found in {log_dir}")
else:
    full_log_content = ""
    for file in log_files:
        with open(os.path.join(log_dir, file), 'r') as f:
            full_log_content += f.read()

    # an epoch block starts with 'Epoch X' and contains the metrics for that epoch.
    # split the entire log by this pattern.
    epoch_blocks = re.split(r'Epoch \d+', full_log_content)

    for block in epoch_blocks:
        if not block.strip():
            continue

        # 1. Extract Dice Scores ---
        train_loss_match = re.search(r'train_loss\s+([-\d.]+)', block)
        pseudo_dice_match = re.search(r'Pseudo dice\s+\[(.*?)\]', block)

        if train_loss_match and pseudo_dice_match:
            train_dice.append(float(train_loss_match.group(1)) * -1)
            dice_values_str = pseudo_dice_match.group(1)
            dice_scores = [float(d) for d in re.findall(r'(\d+\.\d+)', dice_values_str)]
            if dice_scores:
                val_dice.append(np.mean(dice_scores))

        # extract Epoch Duration 
        duration_match = re.search(r'Epoch time: ([\d.]+) s', block)
        if duration_match:
            epoch_durations.append(float(duration_match.group(1)))

    # dice scores
    if train_dice and val_dice:
        plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path_dice = os.path.join(plot_dir, "dice_plot.png")

        plt.figure(figsize=(12, 7))
        plt.plot(train_dice, label='Train Dice (from -loss)', linewidth=2, color='royalblue')
        plt.plot(val_dice, label='Validation Dice (mean pseudo dice)', linewidth=2, color='darkorange')
        
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Dice Score", fontsize=14)
        plt.title("Train vs. Validation Dice Over Epochs", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(plot_path_dice)
        plt.close()

        print(f"Dice plot saved to: {plot_path_dice}")
    else:
        print("Could not extract any Dice scores.")

    # plotting: epoch duration 
    if epoch_durations:
        plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path_duration = os.path.join(plot_dir, "epoch_duration_plot.png")

        plt.figure(figsize=(12, 7))
        plt.plot(epoch_durations, label='Epoch Duration', linewidth=2, color='green', marker='o', markersize=4, linestyle='-')
        
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Duration (seconds)", fontsize=14)
        plt.title("Epoch Duration Over Time", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(plot_path_duration)
        plt.close()

        print(f"epoch duration plot saved to: {plot_path_duration}")
    else:
        print("could not extract any epoch durations.")