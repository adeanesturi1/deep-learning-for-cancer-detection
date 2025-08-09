import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# --- Data Loading and Processing (same as before) ---
# BCBM results
bcbm_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/per_patient_analysis/dice_with_hausdorff.csv"
bcbm_df = pd.read_csv(bcbm_path)
bcbm_dice = bcbm_df['Dice'].mean()

# BraTS results
brats_dice_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/dice_per_class.csv"
brats_dice_df = pd.read_csv(brats_dice_path)
brats_dice = brats_dice_df[['dice_wt', 'dice_tc', 'dice_et']].mean().mean()

# Adversarial BraTS results
adv_path = "/sharedscratch/an252/cancerdetectiondataset/brats_attacks_multi_run_v2/adversarial_attack_summary.csv"
adv_df = pd.read_csv(adv_path)
adv_dice = adv_df[['dice_adv_ET', 'dice_adv_TC', 'dice_adv_WT']].mean().mean()
dice_orig = adv_df[['dice_orig_ET', 'dice_orig_TC', 'dice_orig_WT']].mean().mean()
dice_drop = dice_orig - adv_dice

# --- Visualization Setup ---
labels = ["BraTS (Train/Val)", "BCBM (Fine-tune/Test)", "Adversarial BraTS"]
modalities = [["T1", "T1ce", "T2", "FLAIR"], ["T1ce", "FLAIR"], ["FGSM", "PGD", "Gaussian"]]
colors = ["#66c2a5", "#fc8d62", "#8da0cb"]

dice_scores = [brats_dice, bcbm_dice, adv_dice]
dice_drop_scores = [0, 0, dice_drop]  # Only adversarial has drop
max_drop = max(dice_drop_scores) if max(dice_drop_scores) > 0 else 1
scaled_drop = [val / max_drop for val in dice_drop_scores]

fig, ax = plt.subplots(figsize=(14, 8))
positions = [(0, 1), (1.5, 1), (3, 1)] # Increased spacing for clarity

for i, (x, y) in enumerate(positions):
    # Dataset node (using squares instead of circles)
    ax.add_patch(mpatches.Rectangle((x - 0.25, y - 0.15), 0.5, 0.3, color='lightgray', ec='black', lw=1.5))
    ax.text(x, y, labels[i], ha='center', va='center', fontsize=12, weight='bold')

    # Modality/attack bars below the node
    for j, mod in enumerate(modalities[i]):
        ax.bar(x - 0.3 + j*0.2, 0.6, width=0.15, color=colors[i], ec='black', alpha=0.7)
        ax.text(x - 0.3 + j*0.2, 0.65, mod, ha='center', fontsize=8, rotation=45)

    # Dice & Dice Drop bars
    ax.bar(x - 0.1, dice_scores[i], width=0.08, color="#4daf4a", label="Dice" if i==0 else "")
    ax.text(x - 0.1, dice_scores[i] + 0.02, f"{dice_scores[i]:.2f}", ha='center', fontsize=9)

    ax.bar(x + 0.1, scaled_drop[i], width=0.08, color="#e41a1c", label="Dice Drop" if i==0 else "")
    ax.text(x + 0.1, scaled_drop[i] + 0.02, f"{dice_drop_scores[i]:.2f}", ha='center', fontsize=9)

    if i == 0: # Add labels only once to avoid clutter
        ax.text(x - 0.1, -0.05, "Dice", ha='center', fontsize=9)
        ax.text(x + 0.1, -0.05, "Dice Drop", ha='center', fontsize=9)

# --- Arrows and Labels ---
arrow_props = dict(arrowstyle="->", linewidth=2.5, color="black")
curved_arrow_props = dict(arrowstyle="->", linewidth=2.5, color="black", connectionstyle="arc3,rad=-0.3")

# Arrow 1: BraTS → BCBM (Fine-tune)
ax.annotate("", xy=(positions[1][0] - 0.3, 1), xytext=(positions[0][0] + 0.3, 1), arrowprops=arrow_props)
ax.text(0.75, 1.08, "Fine-tune", ha='center', va='center', fontsize=11, weight='semibold', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0))

# Arrow 2: BraTS → Adversarial BraTS (Attack) - Curved arrow
ax.annotate("", xy=(positions[2][0] - 0.3, 1), xytext=(positions[0][0] + 0.3, 1), arrowprops=curved_arrow_props)
ax.text(1.5, 0.75, "Attack (FGSM/PGD)", ha='center', va='center', fontsize=11, weight='semibold')

# --- Final Plotting Adjustments ---
legend_patches = [
    mpatches.Patch(facecolor='lightgray', ec='black', label='Dataset/Task'),
    mpatches.Patch(color=colors[0], label='BraTS Modalities'),
    mpatches.Patch(color=colors[1], label='BCBM Modalities'),
    mpatches.Patch(color=colors[2], label='Adversarial Attacks'),
    mpatches.Patch(color='#4daf4a', label='Dice Score (Higher is better)'),
    mpatches.Patch(color='#e41a1c', label='Dice Drop (vs. Original)')
]
plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=10)

ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.1, 1.5)
ax.axis('off')

plt.title("Segmentation Performance: From Standard Training to Domain Shift and Adversarial Attacks 🎯", fontsize=16, weight='bold', pad=20)
plt.tight_layout()

plt.savefig("/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/figures/segmentation_pipeline_overview_v2.png", dpi=300, bbox_inches='tight')
plt.show()