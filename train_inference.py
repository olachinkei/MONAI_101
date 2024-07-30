import os
import numpy as np
from tqdm.auto import tqdm
import wandb
import yaml

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

def main():
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    #wandb.require("core")

    with wandb.init(entity="wandb-healthcare",
                    project="MONAI_101",
                    config = config, 
                    job_type= config["mode"]) as run:
        config = wandb.config
        
    
        seed = config.seed
        set_determinism(seed=seed) 

        # Create directories
        #os.makedirs(config.dataset_dir_train, exist_ok=True)
        #os.makedirs(config.dataset_dir_val, exist_ok=True)
        #os.makedirs(config.checkpoint_dir, exist_ok=True)

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
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0)
                ] +
            ([RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.5, 1.5))] if config.data_RandAdjustContrastd else []) +
            ([RandGaussianNoised(keys="image", prob=0.5, std=0.01)] if config.data_RandGaussianNoised else [])
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
        
        if config.data_use_artifacts:
            artifact = run.use_artifact(config.artifact_path)
            artifact_dir = artifact.download()
            config.dataset_dir_train = artifact_dir + "/Task01_BrainTumour_train/"
            print(config.dataset_dir_train)
            config.dataset_dir_val = artifact_dir + "/Task01_BrainTumour_val/"
            

        train_dataset = DecathlonDataset(
            root_dir=config.dataset_dir_train,
            task="Task01_BrainTumour",
            transform=val_transform,
            section="training",
            download=False,
            cache_rate=0.0,
            num_workers=4,
        )
        val_dataset = DecathlonDataset(
            root_dir=config.dataset_dir_val,
            task="Task01_BrainTumour",
            transform=val_transform,
            section="validation",
            download=False,
            cache_rate=0.0,
            num_workers=4,
        )
        ###################################################
        # Loading the Data
        ###################################################
        # We create the PyTorch dataloaders for loading the data from the datasets. Note that before creating the dataloaders, 
        # we set the transform for train_dataset to train_transform to preprocess and transform the data for training.
        # apply train_transforms to the training dataset
        train_dataset.transform = train_transform

        # create the train_loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        # create the val_loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        ###################################################
        # Creating the Model, Loss, and Optimizer
        ###################################################
        #In this tutorial we will be training a SegResNet model based on the paper 3D MRI brain tumor segmentation using autoencoder regularization. 
        # We create the SegResNet model that comes implemented as a PyTorch Module as part of the monai.networks API. 
        # We also create our optimizer and learning rate scheduler.

        device = torch.device("cuda:0")

        # create model
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        ).to(device)

        # create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            float(config.initial_learning_rate),
            weight_decay=float(config.weight_decay),
        )

        # create learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.max_train_epochs
                )

        loss_function = DiceLoss(
            smooth_nr=config.dice_loss_smoothen_numerator,
            smooth_dr=config.dice_loss_smoothen_denominator,
            squared_pred=config.dice_loss_squared_prediction,
            to_onehot_y=config.dice_loss_target_onehot,
            sigmoid=config.dice_loss_apply_sigmoid,
        )

        dice_metric = DiceMetric(include_background=True, reduction="mean")
        dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # use automatic mixed-precision to accelerate training
        scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True

        def inference(model, input):
            def _compute(input):
                return sliding_window_inference(
                    inputs=input,
                    roi_size=(240, 240, 160),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,
                )

            with torch.cuda.amp.autocast():
                return _compute(input)

        ###################################################
        # Training and Validation
        ###################################################
        if config.mode=="train":
            # Before we start training, let us define some metric properties which will later be logged with wandb.log() 
            # for tracking our training and validation experiments.

            wandb.define_metric("epoch/epoch_step")
            wandb.define_metric("epoch/*", step_metric="epoch/epoch_step")
            wandb.define_metric("batch/batch_step")
            wandb.define_metric("batch/*", step_metric="batch/batch_step")
            wandb.define_metric("validation/validation_step")
            wandb.define_metric("validation/*", step_metric="validation/validation_step")

            batch_step = 0
            validation_step = 0
            metric_values = []
            metric_values_tumor_core = []
            metric_values_whole_tumor = []
            metric_values_enhanced_tumor = []


            ###################################################
            # Execute Standard PyTorch Training Loop
            ###################################################
            # Define a W&B Artifact object
            epoch_progress_bar = tqdm(range(config.max_train_epochs), desc="Training:")

            for epoch in epoch_progress_bar:
                model.train()
                epoch_loss = 0

                total_batch_steps = len(train_dataset) // train_loader.batch_size
                batch_progress_bar = tqdm(train_loader, total=total_batch_steps, leave=False)
                
                # Training Step
                for batch_data in batch_progress_bar:
                    inputs, labels = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                    )
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = loss_function(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item()
                    batch_progress_bar.set_description(f"train_loss: {loss.item():.4f}:")
                    ## Log batch-wise training loss to W&B
                    wandb.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
                    batch_step += 1

                lr_scheduler.step()
                epoch_loss /= total_batch_steps
                ## Log batch-wise training loss and learning rate to W&B
                wandb.log(
                    {
                        "epoch/epoch_step": epoch,
                        "epoch/mean_train_loss": epoch_loss,
                        "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
                    }
                )
                epoch_progress_bar.set_description(f"Training: train_loss: {epoch_loss:.4f}:")

                # Validation and model checkpointing
                if (epoch + 1) % config.validation_intervals == 0:
                    model.eval()
                    with torch.no_grad():
                        for val_data in val_loader:
                            val_inputs, val_labels = (
                                val_data["image"].to(device),
                                val_data["label"].to(device),
                            )
                            val_outputs = inference(model, val_inputs)
                            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                            dice_metric(y_pred=val_outputs, y=val_labels)
                            dice_metric_batch(y_pred=val_outputs, y=val_labels)

                        metric_values.append(dice_metric.aggregate().item())
                        metric_batch = dice_metric_batch.aggregate()
                        metric_values_tumor_core.append(metric_batch[0].item())
                        metric_values_whole_tumor.append(metric_batch[1].item())
                        metric_values_enhanced_tumor.append(metric_batch[2].item())
                        dice_metric.reset()
                        dice_metric_batch.reset()

                        checkpoint_path = os.path.join(config.checkpoint_dir, "model.pth")
                        torch.save(model.state_dict(), checkpoint_path)
                        
                        # Log and versison model checkpoints using W&B artifacts.
                        artifact = wandb.Artifact(
                            name=f"{wandb.run.id}-checkpoint", type="model"
                        )
                        artifact.add_file(local_path=checkpoint_path)
                        wandb.log_artifact(artifact, aliases=[f"epoch_{epoch}"])

                        # Log validation metrics to W&B dashboard.
                        wandb.log(
                            {
                                "validation/validation_step": validation_step,
                                "validation/mean_dice": metric_values[-1],
                                "validation/mean_dice_tumor_core": metric_values_tumor_core[-1],
                                "validation/mean_dice_whole_tumor": metric_values_whole_tumor[-1],
                                "validation/mean_dice_enhanced_tumor": metric_values_enhanced_tumor[-1],
                            }
                        )
                        validation_step += 1

        ###################################################
        # Inference
        ###################################################
        elif config.mode=="inference":
            model_artifact = wandb.use_artifact(
                config.model_artifact_path,
                type="model",
            )
            model_artifact_dir = model_artifact.download()
            model.load_state_dict(torch.load(os.path.join(model_artifact_dir, "model.pth")))
            model.eval()

            def log_predictions_into_tables(
                sample_image: np.array,
                sample_label: np.array,
                predicted_label: np.array,
                split: str = None,
                data_idx: int = None,
                table: wandb.Table = None,
            ):
                num_channels, _, _, num_slices = sample_image.shape
                
                # Initialize DiceMetric
                dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
                
                # Convert numpy arrays to PyTorch tensors and add batch dimension
                sample_label_tensor = torch.from_numpy(sample_label).unsqueeze(0)
                predicted_label_tensor = torch.from_numpy(predicted_label).unsqueeze(0)
                
                with tqdm(total=num_slices, leave=False) as progress_bar:
                    for slice_idx in range(num_slices):
                        # Calculate Dice scores for each class on the current slice
                        dice_tc = dice_metric(y_pred=predicted_label_tensor[:, 0:1, :, :, slice_idx:slice_idx+1],
                                            y=sample_label_tensor[:, 0:1, :, :, slice_idx:slice_idx+1]).item()
                        dice_wt = dice_metric(y_pred=predicted_label_tensor[:, 1:2, :, :, slice_idx:slice_idx+1],
                                            y=sample_label_tensor[:, 1:2, :, :, slice_idx:slice_idx+1]).item()
                        dice_et = dice_metric(y_pred=predicted_label_tensor[:, 2:3, :, :, slice_idx:slice_idx+1],
                                            y=sample_label_tensor[:, 2:3, :, :, slice_idx:slice_idx+1]).item()
                        dice_avg = (dice_tc + dice_wt + dice_et) / 3

                        wandb_images = []
                        for channel_idx in range(num_channels):
                            # Create a single 2D mask array for ground-truth and predictions
                            ground_truth_mask = np.zeros_like(sample_label[0, :, :, slice_idx])
                            prediction_mask = np.zeros_like(predicted_label[0, :, :, slice_idx])
                            
                            # Assigning labels directly
                            ground_truth_mask = np.maximum(np.maximum(sample_label[0, :, :, slice_idx], 
                                                                    sample_label[1, :, :, slice_idx]*2), 
                                                                    sample_label[2, :, :, slice_idx]*3)
                            prediction_mask = np.maximum(np.maximum(predicted_label[0, :, :, slice_idx] * 4, 
                                                                    predicted_label[1, :, :, slice_idx] * 5), 
                                                                    predicted_label[2, :, :, slice_idx] * 6)
                            
                            masks = {
                                "ground-truth": {"mask_data": ground_truth_mask, "class_labels": {1: "Tumor Core", 2: "Whole Tumor", 3: "Enhancing Tumor"}},
                                "prediction": {"mask_data": prediction_mask, "class_labels": {4: "Tumor Core", 5: "Whole Tumor", 6: "Enhancing Tumor"}}
                            }
                            
                            wandb_images.append(
                                wandb.Image(
                                    sample_image[channel_idx, :, :, slice_idx],
                                    masks=masks
                                )
                            )
                        table.add_data(split, data_idx, slice_idx, dice_tc, dice_wt, dice_et, dice_avg, *wandb_images)
                        progress_bar.update(1)
                return table

            # create the prediction table
            prediction_table = wandb.Table(
                columns=[
                    "Split",
                    "Data Index",
                    "Slice Index",
                    "Dice_Tumor_Core",
                    "Dice_Whole_Tumor",
                    "Dice_Enhancing_Tumor",
                    "Dice_Average",
                    "Image-Channel-0",
                    "Image-Channel-1",
                    "Image-Channel-2",
                    "Image-Channel-3",
                ]
            )

            # Perform inference and visualization
            with torch.no_grad():
                config.max_prediction_images_visualized
                max_samples = (
                    min(config.max_prediction_images_visualized, len(val_dataset))
                    if config.max_prediction_images_visualized > 0
                    else len(val_dataset)
                )
                progress_bar = tqdm(
                    enumerate(val_dataset[:max_samples]),
                    total=max_samples,
                    desc="Generating Predictions:",
                )
                for data_idx, sample in progress_bar:
                    val_input = sample["image"].unsqueeze(0).to(device)
                    val_output = inference(model, val_input)
                    val_output = post_trans(val_output[0])
                    prediction_table = log_predictions_into_tables(
                        sample_image=sample["image"].cpu().numpy(),
                        sample_label=sample["label"].cpu().numpy(),
                        predicted_label=val_output.cpu().numpy(),
                        data_idx=data_idx,
                        split="validation",
                        table=prediction_table,
                    )

                wandb.log({"Predictions/Tumor-Segmentation-Data": prediction_table})

if __name__ == "__main__":
    main()