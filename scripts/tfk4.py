#!/usr/bin/env python3
import os
import pandas as pd
import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

class PropertiesTransform:
    def __init__(self, label_map):
        """
        label_map: dict mapping case_id -> 0/1
        """
        self.label_map = label_map

    def __call__(self, **sample):
        """
        sample comes in with keys 'data', 'target', 'keys', ...
        we inject a 'properties' entry which is a list of dicts for each case
        """
        batch_case_ids = sample.get("keys", [])
        props = []
        for cid in batch_case_ids:
            # our metadata uses IDs like "BCBM_0001"
            label = self.label_map.get(cid, None)
            props.append({"case_id": cid, "HER2_status": label})
        sample["properties"] = props
        return sample

class nnUNetTrainer_MultiTaskSimple(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda"), **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, device, **kwargs)

        # 1) load your metadata CSV
        metadata_csv = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/bcbm_metadata.csv"
        df = pd.read_csv(metadata_csv)
        df = df[df["HER2_Status"].isin(["+", "-"])].copy()
        df["HER2_Status"] = df["HER2_Status"].map({"+": 1, "-": 0})
        # make a dict: "BCBM_0001" -> 0/1
        self.her2_label_map = dict(zip(df["nnUNet_ID"], df["HER2_Status"]))

        # 2) classification loss
        self.cls_loss_fn = nn.BCEWithLogitsLoss()

        # 3) a CSV logger
        self.log_path = os.path.join(self.output_folder, "training_logs.csv")
        # write header if not exists
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("epoch,iter,seg_loss,cls_loss,total_loss\n")

    def build_network(self):
        """
        After nnUNet builds its U-Net, we attach:
          nn.AdaptiveAvgPool3d(1) → flatten → Linear(bottleneck_ch, 1)
        """
        super().build_network()

        # find bottleneck channels:
        enc = self.network.encoder
        # encoder.stages is a list, last stage's output_channels is our bottleneck
        bottleneck_ch = enc.stages[-1].output_channels

        # classification head
        self.network.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 1)
        )
        # push to correct device
        self.network.cls_head.to(self.device)

        # now monkey-patch forward() to return (seg_logits, cls_logits)
        orig_forward = self.network.forward
        def multi_forward(x):
            # run encoder manually
            skips = enc(x)
            bottleneck = skips[-1]
            seg_logits = self.network.decoder(skips)
            cls_logits = self.network.cls_head(bottleneck)
            return seg_logits, cls_logits
        self.network.forward = multi_forward

    def get_dataloaders(self):
        """
        Wrap the standard data loaders to inject our HER2 labels into 'properties'.
        """
        dl_tr, dl_val = super().get_dataloaders()

        # wrap train loader
        tr_transform = PropertiesTransform(self.her2_label_map)
        dl_tr = NonDetMultiThreadedAugmenter(
            data_loader=dl_tr,
            transform=tr_transform,
            num_processes=2,
            num_cached=2,
            pin_memory=self.device.type == "cuda"
        )

        # wrap val loader
        val_transform = PropertiesTransform(self.her2_label_map)
        dl_val = NonDetMultiThreadedAugmenter(
            data_loader=dl_val,
            transform=val_transform,
            num_processes=1,
            num_cached=1,
            pin_memory=self.device.type == "cuda"
        )

        # prime them
        _ = next(dl_tr)
        _ = next(dl_val)
        return dl_tr, dl_val

    def train_step(self, batch):
        """
        batch is a dict with:
          batch["data"], batch["target"], batch["properties"] (list of dicts)
        """
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        props = batch["properties"]

        self.network.train()
        self.optimizer.zero_grad()

        seg_logits, cls_logits = self.network(data)
        # segmentation
        seg_loss = self.loss(seg_logits, target)

        # classification: gather only those with labels
        cls_labels = []
        cls_preds  = []
        for i, p in enumerate(props):
            lbl = p["HER2_status"]
            if lbl is not None:
                cls_labels.append(lbl)
                cls_preds.append(cls_logits[i])

        if cls_preds:
            cls_preds  = torch.stack(cls_preds).squeeze(1)
            cls_labels = torch.tensor(cls_labels, dtype=torch.float32, device=cls_preds.device)
            cls_loss   = self.cls_loss_fn(cls_preds, cls_labels)
        else:
            cls_loss = torch.tensor(0.0, device=self.device)

        total_loss = seg_loss + 0.5 * cls_loss
        total_loss.backward()
        self.optimizer.step()

        # log to CSV
        it = self.current_iteration  # provided by nnUNetTrainer
        with open(self.log_path, "a") as f:
            f.write(f"{self.epoch},{it},{seg_loss.item():.6f},{cls_loss.item():.6f},{total_loss.item():.6f}\n")

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "seg_loss": seg_loss.detach().cpu().numpy(),
            "cls_loss": cls_loss.detach().cpu().numpy()
        }
