import os

import numpy as np
from tqdm.auto import tqdm
import wandb

from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.config import print_config
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandGaussianNoised,
)
from monai.utils import set_determinism
import torch

print_config()

config = {
    "seed": 0,
    "roi_size": [224, 224, 144],
    "batch_size": 1,
    "num_workers": 4,
    "max_train_images_visualized": 20,
    "max_val_images_visualized": 20,
    "dice_loss_smoothen_numerator": 0,
    "dice_loss_smoothen_denominator": 1e-5,
    "dice_loss_squared_prediction": True,
    "dice_loss_target_onehot": False,
    "dice_loss_apply_sigmoid": True,
    "initial_learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "max_train_epochs": 50,
    "validation_intervals": 1,
    "dataset_dir_train": "./dataset/train",
    "dataset_dir_val": "./dataset/val",
    "checkpoint_dir": "./checkpoints",
    "inference_roi_size": (128, 128, 64),
    "max_prediction_images_visualized": 20
}

wandb.require("core")
with wandb.init(entity="wandb-healthcare",
                project="MONAI_101",
                config = config, 
                name="data_prep",
                job_type="data_prep") as run:
    config = wandb.config

    seed = config.seed
    set_determinism(seed=seed) 

    # Create directories
    #os.makedirs(config.dataset_dir_train, exist_ok=False)
    #os.makedirs(config.dataset_dir_val, exist_ok=False)
    #os.makedirs(config.checkpoint_dir, exist_ok=False)

    ###################################################
    # Data Loading and Transformation
    ###################################################
    # Here we use the monai.transforms API to create a custom transform 
    # that converts the multi-classes labels into multi-labels segmentation task in one-hot format.
    class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
        """
        Convert labels to multi channels based on brats classes:
        label 1 is the peritumoral edema
        label 2 is the GD-enhancing tumor
        label 3 is the necrotic and non-enhancing tumor core
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor).

        Reference: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

        """
        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                result = []
                # merge label 2 and label 3 to construct TC
                result.append(torch.logical_or(d[key] == 2, d[key] == 3))
                # merge labels 1, 2 and 3 to construct WT
                result.append(
                    torch.logical_or(
                        torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                    )
                )
                # label 2 is ET
                result.append(d[key] == 2)
                d[key] = torch.stack(result, axis=0).float()
            return d
    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=config.roi_size, random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.5, 1.5)),
            RandGaussianNoised(keys="image", prob=0.5, std=0.01),
        ]
    )

    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    ###################################################
    # The Dataset 
    ###################################################
    # The dataset that we will use for this experiment comes from http://medicaldecathlon.com/. 
    # We will use Multimodal multisite MRI data (FLAIR, T1w, T1gd, T2w) to segment Gliomas, necrotic/active tumour, and oedema.
    # The dataset consists of 750 4D volumes (484 Training + 266 Testing).
    # We will use the DecathlonDataset to automatically download and extract the dataset. 
    # It inherits MONAI CacheDataset which enables us to set cache_num=N to cache N items for training and use the default args to cache all the items for validation, depending on your memory size.
    #
    # Note: Instead of applying the train_transform to the train_dataset, 
    # we have applied val_transform to both the training and validation datasets. 
    # This is because, before training, we would be visualizing samples from both the splits of the dataset.
    

    train_dataset = DecathlonDataset(
        root_dir=config.dataset_dir_train,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=4,
    )
    val_dataset = DecathlonDataset(
        root_dir=config.dataset_dir_val,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=True,
        cache_rate=0.0,
        num_workers=4,
    )
    
    artifact = wandb.Artifact(
        name="DecathlonDataset_Task01_BrainTumour",
        type="dataset"
    )
    
    artifact.add_dir(
        local_path=config.dataset_dir_train,
        name="Task01_BrainTumour_train")
    artifact.add_dir(
        local_path=config.dataset_dir_val,
        name="Task01_BrainTumour_val")
    run.log_artifact(artifact)

    ###################################################
    # Visualizing the Dataset
    ###################################################
    # Weights & Biases supports images, video, audio, and more. 
    # Log rich media to explore our results and visually compare our runs, models, and datasets.
    # We would be using the segmentation mask overlay system to visualize our data volumes. 
    # To log segmentation masks in tables, we will need to provide a `wandb.Image`` object for each row in the table.
    # An example is provided in the Code snippet below:

    def log_data_samples_into_tables(
        sample_image: np.array,
        sample_label: np.array,
        split: str = None,
        data_idx: int = None,
        table: wandb.Table = None,
    ):
        num_channels, _, _, num_slices = sample_image.shape
        with tqdm(total=num_slices, leave=False) as progress_bar:
            for slice_idx in range(num_slices):
                ground_truth_wandb_images = []
                for channel_idx in range(num_channels):
                    ground_truth_wandb_images.append(
                        wandb.Image(
                            sample_image[channel_idx, :, :, slice_idx],
                            masks={
                                "ground-truth/Tumor-Core": {
                                    "mask_data": sample_label[0, :, :, slice_idx],
                                    "class_labels": {0: "background", 1: "Tumor Core"},
                                },
                                "ground-truth/Whole-Tumor": {
                                    "mask_data": sample_label[1, :, :, slice_idx] * 2,
                                    "class_labels": {0: "background", 2: "Whole Tumor"},
                                },
                                "ground-truth/Enhancing-Tumor": {
                                    "mask_data": sample_label[2, :, :, slice_idx] * 3,
                                    "class_labels": {0: "background", 3: "Enhancing Tumor"},
                                },
                            },
                        )
                    )
                table.add_data(split, data_idx, slice_idx, *ground_truth_wandb_images)
                progress_bar.update(1)
        return table

    table = wandb.Table(
        columns=[
            "Split",
            "Data Index",
            "Slice Index",
            "Image-Channel-0",
            "Image-Channel-1",
            "Image-Channel-2",
            "Image-Channel-3",
        ]
    )
    # Then we loop over the train_dataset and val_dataset respectively to generate
    # the visualizations for the data samples and populate the rows of the table which we would log to our dashboard.

    # Generate visualizations for train_dataset
    max_samples = (
        min(config.max_train_images_visualized, len(train_dataset))
        if config.max_train_images_visualized > 0
        else len(train_dataset)
    )
    progress_bar = tqdm(
        enumerate(train_dataset[:max_samples]),
        total=max_samples,
        desc="Generating Train Dataset Visualizations:",
    )
    for data_idx, sample in progress_bar:
        sample_image = sample["image"].detach().cpu().numpy()
        sample_label = sample["label"].detach().cpu().numpy()
        table = log_data_samples_into_tables(
            sample_image,
            sample_label,
            split="train",
            data_idx=data_idx,
            table=table,
        )

    # Generate visualizations for val_dataset
    max_samples = (
        min(config.max_val_images_visualized, len(val_dataset))
        if config.max_val_images_visualized > 0
        else len(val_dataset)
    )
    progress_bar = tqdm(
        enumerate(val_dataset[:max_samples]),
        total=max_samples,
        desc="Generating Validation Dataset Visualizations:",
    )
    for data_idx, sample in progress_bar:
        sample_image = sample["image"].detach().cpu().numpy()
        sample_label = sample["label"].detach().cpu().numpy()
        table = log_data_samples_into_tables(
            sample_image,
            sample_label,
            split="val",
            data_idx=data_idx,
            table=table,
        )

    # Log the table to your dashboard
    wandb.log({"Tumor-Segmentation-Data": table})

