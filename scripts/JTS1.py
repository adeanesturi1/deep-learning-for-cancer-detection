import os
import json
import matplotlib.pyplot as plt


log_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3"
json_path = os.path.join(log_dir, 'debug.json')

if not os.path.exists(json_path):
    print(f"Error: debug.json not found in {log_dir}")
else:
    with open(json_path, 'r') as f:
        data = json.load(f)

    # extract data from the JSON file
    # training loss in debug.json is a combined loss (e.g., Dice + Cross-Entropy).
    train_losses = data.get('train_losses')
    
    # the 'val_eval_criterion' key holds the validation Dice scores.
    val_dice = data.get('val_eval_criterion')
    
    # the 'epoch_times' key holds the duration of each epoch.
    epoch_durations = data.get('epoch_times')
    
    # check if data was successfully extracted before proceeding
    if not all([train_losses, val_dice, epoch_durations]):
        print("Error: One or more required keys ('train_losses', 'val_eval_criterion', 'epoch_times') were not found in debug.json.")
    else:
        epochs = list(range(1, len(train_losses) + 1))
        plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path_dice = os.path.join(plot_dir, "loss_dice_plot.png")
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # plot training loss on the primary y-axis
        color = 'royalblue'
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Training Loss', fontsize=14, color=color)
        ax1.plot(epochs, train_losses, label='Training Loss', linewidth=2, color=color)
        ax1.tick_params(axis='y', labelcolor=color)        
        ax2 = ax1.twinx()
        color = 'darkorange'
        ax2.set_ylabel('Validation Dice Score', fontsize=14, color=color)
        ax2.plot(epochs, val_dice, label='Validation Dice', linewidth=2, color=color, marker='.', markersize=5)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title("Training Loss vs. Validation Dice Over Epochs", fontsize=16)
        fig.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(plot_path_dice)
        plt.close()

        print(f"Loss vs. Dice plot saved to: {plot_path_dice}")

        # plotting: epoch duration ---
        plot_path_duration = os.path.join(plot_dir, "epoch_duration_plot.png")

        plt.figure(figsize=(12, 7))
        plt.plot(epochs, epoch_durations, label='Epoch Duration', linewidth=2, color='green', marker='o', markersize=4, linestyle='-')
        
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Duration (seconds)", fontsize=14)
        plt.title("Epoch Duration Over Time", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path_duration)
        plt.close()

        print(f"Epoch duration plot saved to: {plot_path_duration}")