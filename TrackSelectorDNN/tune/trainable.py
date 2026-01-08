"""
Module defining the Ray Tune trainable for TrackClassifier training.
"""

import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ray.air import session
from ray.train import Checkpoint

from TrackSelectorDNN.configs.schema import Config, load_config
from TrackSelectorDNN.data_manager.dataset import FeatureBundle
from TrackSelectorDNN.data_manager.dataset_factory import get_dataset
from TrackSelectorDNN.models.factory import build_model
from TrackSelectorDNN.train.optim_factory import build_optimizer, build_scheduler
from TrackSelectorDNN.tune.utils_logging import (
    CSVLogger,
    create_run_dir,
    save_config,
    save_model_summary,
)

# ---------------------------
# Utility functions
# ---------------------------


def mirror_inputs(
        features: FeatureBundle, 
        idx_sym_hit_features=None,
        idx_sym_track_features=None,
        idx_sym_preselect_features=None
) -> FeatureBundle:
    """
    Apply symmetric mirroring to specified features in FeatureBundle.

    Args:
        features (FeatureBundle): Input features
        idx_sym_hit_features (list[int] or None): Indices of hit features to mirror
        idx_sym_track_features (list[int] or None): Indices of track features to mirror
        idx_sym_preselect_features (list[int] or None): Indices of preselect features

    Returns:
        FeatureBundle: Mirrored features
    """
    hit_features = features.hit_features.clone() if features.hit_features is not None else None
    track_features = features.track_features.clone() if features.track_features is not None else None
    preselect_features = features.preselect_features.clone() if features.preselect_features is not None else None

    if (idx_sym_hit_features is not None) and (features.hit_features is not None):
        for idx in idx_sym_hit_features:
            hit_features[:,:,idx]*=-1

    if (idx_sym_track_features is not None) and (features.track_features is not None):  
        for idx in idx_sym_track_features:
            track_features[:,idx]*=-1

    if (idx_sym_preselect_features is not None) and (features.preselect_features is not None):
        for idx in idx_sym_preselect_features:
            preselect_features[:,idx]*=-1

    return FeatureBundle(
        hit_features=hit_features,
        track_features=track_features,
        preselect_features=preselect_features,  
        mask=features.mask
    )

def train_one_epoch(
        model,
        loader,
        optimizer,
        device,
        idx_sym_hit_features,
        idx_sym_track_features,
        idx_sym_preselect_features,
        lambda_sym,
        w_true,
        w_fake,
        scheduler,
        scheduler_type):
    """
    Train the model for one epoch.

    Supports:
        - Weighted binary cross-entropy loss
        - Symmetry loss regularization
        - Batched training with DataLoader

    Args:
        model (nn.Module): TrackClassifier model
        loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer for gradient updates
        device (str): "cuda" or "cpu"
        idx_sym_hit_features (list[int] or None): Indices of hit features to symmetrtize
        idx_sym_track_features (list[int] or None): Indices of track features to symmetrize
        idx_sym_preselect_features (list[int] or None): Indices of preselect features to symmetrize
        lambda_sym (float or None): Weight for symmetry regularization
        w_true (torch.Tensor or None): Weight for positive labels
        w_fake (torch.Tensor or None): Weight for negative labels
        scheduler: Learning rate scheduler
        scheduler_type: Type of scheduler ("epoch" or "batch")

    Returns:
        float: Average training loss for the epoch
    """

    model.train()
    total_loss = 0
    total_loss_sym = 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        preds = model.forward_bundle(features)

        if (w_true is not None)and (w_fake is not None):
            weight = torch.where(labels==1, w_true, w_fake)
            loss = nn.functional.binary_cross_entropy_with_logits(preds, labels, weight=weight)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(preds, labels)

        if lambda_sym is not None and any([
            idx_sym_hit_features is not None,
            idx_sym_track_features is not None,
            idx_sym_preselect_features is not None
            ]):

            features_mirr = mirror_inputs(features, idx_sym_hit_features, idx_sym_track_features, idx_sym_preselect_features)
            features_mirr = features_mirr.to(device)

            preds_mirr = model.forward_bundle(features_mirr)
            loss_sym = lambda_sym * (torch.sigmoid(preds_mirr) - torch.sigmoid(preds)).pow(2).mean()
            loss += loss_sym
            total_loss_sym += loss_sym.item() * len(labels)

        loss.backward()
        optimizer.step()
        if (scheduler is not None) and (scheduler_type == "batch"):
            scheduler.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset), total_loss_sym / len(loader.dataset)


def validate(
    model,
    loader,
    device,
    idx_sym_hit_features,
    idx_sym_track_features,
    idx_sym_preselect_features,
    lambda_sym,
    w_true,
    w_fake):
    """
    Evaluate the model on a validation dataset.

    Args:
        model (nn.Module): TrackClassifier model
        loader (DataLoader): Validation data loader
        device (str): "cuda" or "cpu"
        idx_sym_hit_features (list[int] or None): Indices of hit features for symmetry
        idx_sym_track_features (list[int] or None): Indices of track features for symmetry
        idx_sym_preselect_features (list[int] or None): Indices of preselect features for symmetry
        lambda_sym (float or None): Weight for symmetry regularization
        w_true (torch.Tensor or None): Weight for positive labels
        w_fake (torch.Tensor or None): Weight for negative labels

    Returns:
        tuple(float, float): Validation loss and accuracy
    """

    model.eval()
    total_loss = 0
    total_loss_sym = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            features: FeatureBundle = batch["features"].to(device)
            labels = batch["labels"].to(device)

            preds = model.forward_bundle(features)
            if (w_true is not None)and (w_fake is not None):
                weight = torch.where(labels==1, w_true, w_fake)
                loss = nn.functional.binary_cross_entropy_with_logits(preds, labels, weight=weight)
            else:    
                loss = nn.functional.binary_cross_entropy_with_logits(preds, labels)

            if lambda_sym is not None and any([
                idx_sym_hit_features is not None,
                idx_sym_track_features is not None,
                idx_sym_preselect_features is not None
            ]):
                features_mirr = mirror_inputs(
                    features,
                    idx_sym_hit_features,
                    idx_sym_track_features,
                    idx_sym_preselect_features
                )

                features_mirr = features_mirr.to(device)

                preds_mirr = model.forward_bundle(features_mirr)
                loss_sym = lambda_sym * (torch.sigmoid(preds_mirr) - torch.sigmoid(preds)).pow(2).mean()
                total_loss_sym += loss_sym.item() * len(labels)

            total_loss += loss.item() * len(labels)
            preds_bin = (preds > 0).float()
            correct += (preds_bin == labels).sum().item()
            n += len(labels)
    return total_loss / n, correct / n, total_loss_sym/n


# ---------------------------
# Ray Tune Trainable
# ---------------------------

def trainable(config):
    """
    Ray Tune trainable function for TrackClassifier training.

    Handles:
        - Configuration parsing
        - Dataset and DataLoader creation
        - Model initialization and optimizer setup
        - Training loop with early stopping
        - Best model checkpointing and metric reporting to Ray

    Args:
        config (dict or ): Hyperparameter and training configuration dictionary
    """
    if isinstance(config, dict):
        config = Config(**config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    patience = config.training.patience
    delta = config.training.delta
    base_checkpoint_directory = config.training.base_checkpoint_directory
    idx_sym_hit_features = config.training.symmetry.idxSymRecHitFeatures
    idx_sym_track_features = config.training.symmetry.idxSymRecoPixelTrackFeatures
    idx_sym_preselect_features = config.training.symmetry.idxSymPreselectFeatures
    lambda_sym = config.training.symmetry.lambda_sym
    w_fake = torch.tensor(config.training.weights.w_fake, device=device)
    w_true = torch.tensor(config.training.weights.w_true, device=device)

    # Trial directory for this run
    trial_name = None
    if session.get_session():
        try:
            trial_name = session.get_trial_id()
        except Exception:
            pass

    run_dir = create_run_dir(
        base_dir=base_checkpoint_directory,
        trial_name=trial_name
    )
    save_config(config, run_dir)

    # Dataset
    train_ds, collate_fn = get_dataset(config, dataset_role="train_path")
    val_ds, _ = get_dataset(config, dataset_role="val_path")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        collate_fn=collate_fn
    )

    # Model
    model = build_model(config.model).to(device)

    save_model_summary(model, run_dir)

    optimizer = build_optimizer(model, config)
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise RuntimeError("train_loader has zero steps_per_epoch (empty dataset)")
    n_epochs = config.training.epochs
    total_steps = n_epochs * steps_per_epoch

    # Three cases:
    # 1. None → no scheduler
    # 2. epoch scheduler
    # 3. batch scheduler
    scheduler, scheduler_type = build_scheduler(optimizer, config, total_steps)
    if scheduler is not None:
        scheduler.last_step = -1

    logger = CSVLogger(run_dir)

    # Initialize best metrics safely
    best_val_loss = float("inf")
    best_metrics = {"val_loss": float("inf"), "val_acc": 0.0, "epoch": 0}
    count_stopping = 0

    # Training loop
    for epoch in range(n_epochs):
        if count_stopping == patience:
            break

        train_loss, train_loss_sym = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            idx_sym_hit_features,
            idx_sym_track_features,
            idx_sym_preselect_features,
            lambda_sym,
            w_true,
            w_fake,
            scheduler,
            scheduler_type
        )

        val_loss, val_acc, val_loss_sym = validate(
            model,
            val_loader,
            device,
            idx_sym_hit_features,
            idx_sym_track_features,
            idx_sym_preselect_features,
            lambda_sym,
            w_true,
            w_fake
        )

        if (scheduler is not None) and (scheduler_type == "epoch"):
            if scheduler_type == "epoch" and config.training.scheduler.name.lower() == "plateau":
                # ReduceLROnPlateau requires the validation loss
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_loss_sym": train_loss_sym,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_loss_sym": val_loss_sym,
            "lr": current_lr,
        }

        print(f"[Epoch {epoch+1}/{n_epochs}] "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        logger.log(metrics)

        # Save best checkpoint only
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics

            # Save lightweight checkpoint directory
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

            with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
                json.dump(best_metrics, f)

            session.report(
                metrics,
                checkpoint=Checkpoint.from_directory(run_dir)
            )
            count_stopping = 0
        if val_loss + delta > best_val_loss :
            count_stopping+=1

    # Final report once training ends
    print("[DEBUG] Final report to Ray:", run_dir)
    with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
        json.dump(best_metrics, f)

    session.report(
        best_metrics,
        checkpoint=Checkpoint.from_directory(run_dir)
    )

if __name__ == "__main__":
    cfg = load_config("base.yaml")
    print(cfg)
    trainable(cfg)
