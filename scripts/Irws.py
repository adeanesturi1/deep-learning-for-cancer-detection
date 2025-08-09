import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# BCBM results
bcbm_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/per_patient_analysis/dice_with_hausdorff.csv"
bcbm_df = pd.read_csv(bcbm_path)
bcbm_dice = bcbm_df['Dice'].mean()
bcbm_hd95 = bcbm_df['HD95'].mean()

# BraTS results
brats_dice_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/dice_per_class.csv"
brats_hd_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/hausdorff_sample.csv"
brats_dice_df = pd.read_csv(brats_dice_path)
brats_hd_df = pd.read_csv(brats_hd_path)

brats_dice = brats_dice_df[['ET', 'TC', 'WT']].mean().mean()
brats_hd95 = brats_hd_df[['ET', 'TC', 'WT']].mean().mean()

# Adversarial BraTS results
adv_path = "/sharedscratch/an252/cancerdetectiondataset/brats_attacks_multi_run_v2/adversarial_attack_summary.csv"
adv_df = pd.read_csv(adv_path)
adv_dice = adv_df[['ET_dice', 'TC_dice', 'WT_dice']].mean().mean()
adv_hd95 = adv_df[['ET_hd95', 'TC_hd95', 'WT_hd95']].mean().mean()

labels = ["BraTS (Train/Val)", "BCBM (Fine-tune/Test)", "Adversarial BraTS"]
modalities = [["T1", "T1ce", "T2", "FLAIR"], ["T1ce", "FLAIR"], ["FGSM", "PGD", "Gaussian"]]
colors = ["#66c2a5", "#fc8d62", "#8da0cb"]

dice_scores = [brats_dice, bcbm_dice, adv_dice]
hd95_scores = [brats_hd95, bcbm_hd95, adv_hd95]
max_hd95 = max(hd95_scores)
scaled_hd95 = [max_hd95 - val for val in hd95_scores]

fig, ax = plt.subplots(figsize=(14, 7))
positions = [(0, 1), (1, 1), (2, 1)]

for i, (x, y) in enumerate(positions):
    # Dataset node
    ax.add_patch(plt.Circle((x, y), 0.2, color='lightgray'))
    ax.text(x, y + 0.3, labels[i], ha='center', fontsize=12, weight='bold')

    # Modality/attack bars
    for j, mod in enumerate(modalities[i]):
        ax.bar(x - 0.3 + j*0.2, 0.1, width=0.1, color=colors[i])
        ax.text(x - 0.3 + j*0.2, 0.15, mod, ha='center', fontsize=8, rotation=45)

    # Dice & HD95 bars
    ax.bar(x - 0.07, dice_scores[i], width=0.05, color="#4daf4a")
    ax.text(x - 0.07, dice_scores[i] + 0.02, f"{dice_scores[i]:.2f}", ha='center', fontsize=8)

    ax.bar(x + 0.07, scaled_hd95[i] / max_hd95, width=0.05, color="#e41a1c")
    ax.text(x + 0.07, scaled_hd95[i] / max_hd95 + 0.02, f"{hd95_scores[i]:.1f}", ha='center', fontsize=8)

    ax.text(x - 0.07, -0.05, "Dice", ha='center', fontsize=9)
    ax.text(x + 0.07, -0.05, "HD95", ha='center', fontsize=9)


legend_patches = [
    mpatches.Patch(color='lightgray', label='Dataset/Task'),
    mpatches.Patch(color=colors[0], label='BraTS modalities'),
    mpatches.Patch(color=colors[1], label='BCBM modalities'),
    mpatches.Patch(color=colors[2], label='Adversarial attacks'),
    mpatches.Patch(color='#4daf4a', label='Dice score'),
    mpatches.Patch(color='#e41a1c', label='HD95 (lower = better)')
]
plt.legend(handles=legend_patches, loc='upper right')

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.1, 1.8)
ax.axis('off')

plt.title("Deep Learning Segmentation: Performance and Robustness Across Domains", fontsize=14)
plt.tight_layout()

plt.savefig("/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/figures/segmentation_pipeline_overview.png", dpi=300, bbox_inches='tight')
plt.show()
