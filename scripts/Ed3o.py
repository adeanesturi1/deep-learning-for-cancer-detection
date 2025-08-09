import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor

def assign_grade(patient_id):
    try:
        num = int(patient_id.split('_')[1])
    except Exception:
        return "Unknown"
    cutoff = 65  # BraTS 2021 has 65 LGG cases
    return "LGG" if num <= cutoff else "HGG"

# Utility functions
def resample_to_1mm(volume, spacing): 
    resize_factors = [s / 1.0 for s in spacing]
    return zoom(volume, resize_factors, order=1)

def z_score_normalize(volume):
    mean = np.mean(volume)
    std = np.std(volume)
    return (volume - mean) / std if std > 0 else volume

def crop_with_bbox(image, padding=15):
    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(image_uint8, (5, 5), 0)
    thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if len(cnts) == 0:
        return image, (0, 0, image.shape[1], image.shape[0])
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    return image[y:y+h+padding, x:x+w+padding], (x, y, w, h)

def save_multimodal_comparison(mod_slices, seg_slice, patient_id, save_dir):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ['T1', 'T1ce', 'T2', 'FLAIR', 'Segmentation']
    for i, (mod, title) in enumerate(zip(mod_slices, titles[:-1])):
        axes[i].imshow(mod, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    axes[4].imshow(mod_slices[0], cmap='gray')
    axes[4].imshow(seg_slice, cmap='Reds', alpha=0.4)
    axes[4].set_title(titles[-1])
    axes[4].axis('off')
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{patient_id}_multimodal.png"))
    plt.close(fig)

modalities = ['t1', 't1ce', 't2', 'flair']
data_dirs = ["/sharedscratch/an252/cancerdetectiondataset/BraTS2021_TrainingSet"]
output_dir = "/sharedscratch/an252/cancerdetectiondataset/output/brats"
os.makedirs(output_dir, exist_ok=True)

def process_single_case(root):
    try:
        patient_id = os.path.basename(root)
        origin = os.path.basename(os.path.dirname(root))
        subgroup = "Glioma"
        grade = assign_grade(patient_id)

        images = {}
        spacing = None
        for mod in modalities:
            path = os.path.join(root, f"{patient_id}_{mod}.nii.gz")
            if not os.path.exists(path):
                return None
            img_nii = nib.load(path)
            img = img_nii.get_fdata()
            spacing = img_nii.header.get_zooms()
            if not np.allclose(spacing, (1.0, 1.0, 1.0)):
                img = resample_to_1mm(img, spacing)
            img = z_score_normalize(img)
            images[mod] = img

        seg_path = os.path.join(root, f"{patient_id}_seg.nii.gz")
        if not os.path.exists(seg_path):
            return None
        seg = nib.load(seg_path).get_fdata()
        if not np.allclose(spacing, (1.0, 1.0, 1.0)):
            seg = resample_to_1mm(seg, spacing)

        mid_idx = images['flair'].shape[2] // 2
        flair_slice = images['flair'][:, :, mid_idx]
        flair_cropped, bbox = crop_with_bbox(flair_slice)
        x, y, w, h = bbox

        slices = {}
        for mod in modalities:
            full_slice = images[mod][:, :, mid_idx]
            slices[mod] = full_slice[y:y+h+15, x:x+w+15]

        seg_slice = seg[:, :, mid_idx]
        seg_slice_cropped = seg_slice[y:y+h+15, x:x+w+15]

        patient_out = os.path.join(output_dir, patient_id)
        os.makedirs(patient_out, exist_ok=True)
        for mod, slice_img in slices.items():
            np.save(os.path.join(patient_out, f"{mod}_slice.npy"), slice_img)
        np.save(os.path.join(patient_out, "seg_slice.npy"), seg_slice_cropped)

        save_multimodal_comparison(list(slices.values()), seg_slice_cropped, patient_id, os.path.join(output_dir, "comparisons"))

        return {
            "PatientID": patient_id,
            "Origin": origin,
            "Subgroup": subgroup,
            "Grade": grade,
            "Voxel Size": spacing,
            "Shape": images['t1'].shape,
            "Min Intensity": float(np.min(images['t1ce'])),
            "Max Intensity": float(np.max(images['t1ce'])),
            "Nonzero Voxels": int(np.sum(seg_slice_cropped > 0))
        }
    except Exception as e:
        print(f"[Error] {root}: {e}")
        return None

def is_valid_patient_folder(folder_path):
    patient_id = os.path.basename(folder_path)
    expected = [f"{patient_id}_{mod}.nii.gz" for mod in modalities + ['seg']]
    return all(os.path.exists(os.path.join(folder_path, f)) for f in expected)

all_entries = []
for base in data_dirs:
    for root, dirs, _ in os.walk(base):
        for d in dirs:
            full_path = os.path.join(root, d)
            if is_valid_patient_folder(full_path):
                all_entries.append(full_path)

print(f"Found {len(all_entries)} valid patient folders.")

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(process_single_case, all_entries), total=len(all_entries), desc="Preprocessing"))

results = [r for r in results if r is not None]
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "preprocessing_metadata.csv"), index=False)

train_val, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.1, random_state=42)
df["Split"] = "None"
df.loc[df["PatientID"].isin(train["PatientID"]), "Split"] = "Train"
df.loc[df["PatientID"].isin(val["PatientID"]), "Split"] = "Validation"
df.loc[df["PatientID"].isin(test["PatientID"]), "Split"] = "Test"
df.to_csv(os.path.join(output_dir, "preprocessing_metadata_with_splits.csv"), index=False)

def plot_and_save(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def plot_voxel_distributions(df):
    fig = plt.figure(figsize=(10, 5))
    sns.histplot(df["Voxel Size"].apply(lambda x: x[0]), kde=True, bins=20)
    plt.title("Voxel Size Distribution")
    plot_and_save(fig, "voxel_distribution.png")

def plot_intensity_distributions(df):
    fig = plt.figure(figsize=(10, 5))
    sns.histplot(df["Max Intensity"], kde=True, color='red', label='Max')
    sns.histplot(df["Min Intensity"], kde=True, color='blue', label='Min')
    plt.title("Intensity Value Distribution")
    plt.legend()
    plot_and_save(fig, "intensity_distribution.png")

def plot_voxels(df):
    fig = plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Grade', y='Nonzero Voxels')
    plt.title("Tumour Voxels by Grade")
    plot_and_save(fig, "tumour_voxels_by_grade.png")

plot_voxel_distributions(df)
plot_intensity_distributions(df)
plot_voxels(df)

print("BraTS preprocessing complete.")
