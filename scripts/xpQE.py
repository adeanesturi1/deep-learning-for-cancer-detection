import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = '/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/dice_summary.csv'

output_dir = '/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/figures'
output_filename = 'dice_summary_visualization.png'
output_path = os.path.join(output_dir, output_filename)

try:
    df = pd.read_csv(file_path)

    # data extraction and renaming 
    # The last two rows of the nnU-Net summary contain the mean and std dev
    # Extract the second to last row for mean scores
    mean_scores = df.iloc[-2]
    # Extract the last row for standard deviation
    std_devs = df.iloc[-1]
    # Dice_fg -> Foreground
    # Dice_1 -> Enhancing Tumour (ET)
    # Dice_2 -> Tumor Core (TC)
    # Dice_3 -> Whole Tumor (WT)
    rename_map = {
        "Dice_fg": "Foreground",
        "Dice_1": "Enhancing Tumour (ET)",
        "Dice_2": "Tumor Core (TC)",
        "Dice_3": "Whole Tumor (WT)"
    }
    
    mean_scores = mean_scores.rename(index=rename_map)
    std_devs = std_devs.rename(index=rename_map)

    # we only want to plot the renamed columns
    plot_labels = list(rename_map.values())
    plot_means = mean_scores[plot_labels]
    plot_stds = std_devs[plot_labels]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(plot_labels, plot_means, yerr=plot_stds,
                  capsize=5, color=plt.cm.viridis(range(len(plot_labels))), alpha=0.8)

    # add titles and labels
    ax.set_title('Dice Scores (with Standard Deviation)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_xlabel('Tumour Subregion', fontsize=12)
    ax.set_ylim(0, 1.05) # Set Y-axis limit to just above 1.0

    # add the mean value text on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', 
                va='bottom', ha='center', fontsize=10, fontweight='bold')

    # improve layout to prevent labels from overlapping
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300)
    
    print(f"Successfully generated and saved the summary plot to: {output_path}")

    print("\nData used for plotting:")
    summary_df = pd.DataFrame({'Mean': plot_means, 'StdDev': plot_stds})
    print(summary_df)

except FileNotFoundError:
    print(f"Error: The input file was not found at the path: {file_path}")
    print("Please make sure the file exists and the path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")

