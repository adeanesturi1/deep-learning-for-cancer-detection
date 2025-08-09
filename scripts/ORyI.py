import json

input_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/dataset.json"
output_path = "/sharedscratch/an252/temp_dataset_fixed.json"
with open(input_path, "r") as f:
    data = json.load(f)

data["labels"] = {str(k): v for k, v in data["labels"].items()}
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"Fully fixed JSON written to: {output_path}")
