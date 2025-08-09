import os
import json

# set base paths
base_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS"
images_dir = os.path.join(base_dir, "imagesTr")
labels_dir = os.path.join(base_dir, "labelsTr")

# updated labels: keys are strings (names), values are integers
labels = {
    "background": 0,
    "edema": 1,
    "non-enhancing tumor core": 2,
    "enhancing tumor": 3
}

# modality/channel names: keys must be strings for valid JSON
modalities = {
    "0": "FLAIR",
    "1": "T1",
    "2": "T1ce",
    "3": "T2"
}

channel_names = modalities.copy()

# build training pairs
training_pairs = []
for fname in sorted(os.listdir(images_dir)):
    if fname.endswith("_0000.nii.gz"):
        case_id = "_".join(fname.split("_")[:2])
        image_path = f"./imagesTr/{case_id}_0000.nii.gz"
        label_path = f"./labelsTr/{case_id}.nii.gz"
        training_pairs.append({
            "image": image_path,
            "label": label_path
        })

# construct the full dataset.json dictionary
dataset_json = {
    "name": "BraTS2021",
    "description": "Brain Tumor Segmentation using the BraTS 2021 dataset. 4 modalities: T1, T1ce, T2, FLAIR.",
    "tensorImageSize": "4D",
    "reference": "https://www.med.upenn.edu/cbica/brats2021/data.html",
    "licence": "CC BY-SA 4.0",
    "release": "1.0",
    "modality": modalities,
    "channel_names": channel_names,
    "labels": labels,
    "file_ending": ".nii.gz",
    "numTraining": len(training_pairs),
    "numTest": 0,
    "training": training_pairs,
    "test": []
}

# write to dataset.json
output_path = os.path.join(base_dir, "dataset.json")
with open(output_path, "w") as f:
    json.dump(dataset_json, f, indent=4)

# confirm
print(f"dataset.json written successfully to: {output_path}")
print("labels:", dataset_json["labels"])
print("types of label values:", [type(v) for v in dataset_json["labels"].values()])
