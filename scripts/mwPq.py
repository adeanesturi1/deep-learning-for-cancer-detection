import os
import json

os.environ['nnUNet_raw'] = '/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/sharedscratch/an252/cancerdetectiondataset/nnUNet_results'

from nnunetv2.paths import nnUNet_results
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer_FrozenEncoderBCBM

print(" Import successful: nnUNetTrainer_FrozenEncoderBCBM")

plans_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans.json"
with open(plans_path, 'r') as f:
    plans = json.load(f)

cfg = plans["configurations"]["3d_fullres"]
arch = cfg["architecture"]["arch_kwargs"]

configuration = {
    "num_input_channels": 4,
    "num_output_classes": 3,
    "unet_class_name": "PlainConvUNet",
    "base_num_features": arch["features_per_stage"][0],
    "num_pool": arch["n_stages"] - 1,
    "conv_per_stage": arch["n_conv_per_stage"][0],
    "conv_op": "Conv3d",
    "norm_op": "InstanceNorm3d",
    "norm_op_kwargs": arch["norm_op_kwargs"],
    "dropout_op": None,
    "dropout_op_kwargs": None,
    "nonlin": "LeakyReLU",
    "nonlin_kwargs": arch["nonlin_kwargs"],
    "deep_supervision": True,
    "pretrainer_name": None
}

dataset_json = {
    "channel_names": {
        "0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"
    },
    "labels": {
        "background": 0, "edema": 1, "non-enhancing": 2, "enhancing": 3
    },
    "file_ending": ".npz"
}
task_name = "Dataset002_BCBM"
trainer_name = "nnUNetTrainer_FrozenEncoderBCBM"
plans_identifier = "nnUNetPlans"
configuration_key = "3d_fullres"
fold = 3

output_folder_base = os.path.join(
    nnUNet_results,
    task_name,
    f"{trainer_name}__{plans_identifier}",
    configuration_key
)

print(f"[DEBUG] plans type: {type(plans)}")
print(f"[DEBUG] configuration keys: {list(configuration.keys())}")
print(f"[DEBUG] dataset_json keys: {list(dataset_json.keys())}")

print("Initializing trainer...")
trainer = nnUNetTrainer_FrozenEncoderBCBM(
    plans=plans,
    configuration="3d_fullres",  
    fold=3,
    dataset_json=dataset_json
)

trainer.output_folder_base = output_folder_base
trainer.output_folder = os.path.join(output_folder_base, f'fold_{fold}')

print("Calling initialize()...")
trainer.initialize(training=True)


print("Trainer setup and freezing test complete.")
