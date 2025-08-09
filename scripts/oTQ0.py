import pandas as pd
import matplotlib.pyplot as plt
import os


file_path = '/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/dice_per_class.csv'
output_dir = '/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/figures'
output_filename = 'dice_score_visualization.png'
output_path = os.path.join(output_dir, output_filename)

try:
    df = pd.read_csv(file_path)
    identifier_column = df.columns[0]
    dice_scores = df.drop(columns=[identifier_column])
    mean_scores = dice_scores.mean()
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(range(len(mean_scores)))
    bars = ax.bar(mean_scores.index, mean_scores.values, color=colors)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', va='bottom', ha='center', fontsize=10)

    ax.set_title('Mean Dice Score per Tumor Class', fontsize=16, fontweight='bold')
    ax.set_xlabel('Tumor Class', fontsize=12)
    ax.set_ylabel('Mean Dice Score', fontsize=12)
    ax.set_ylim(0, 1)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300)
    
    print(f"Successfully generated and saved the plot to: {output_path}")

  

    print("\nMean Dice Scores:")
    print(mean_scores)

except FileNotFoundError:
    print(f"Error: The input file was not found at the path: {file_path}")
    print("Please make sure you are running this script from a machine that has access to this file path.")
except Exception as e:
    print(f"An error occurred: {e}")

