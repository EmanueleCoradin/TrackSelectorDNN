"""
Module to create datasets based on configuration.
"""

from TrackSelectorDNN.data_manager.dataset import (
    TrackDatasetFromFile,
    TrackFullDNNView,
    OriginalDataTrackDNNView,
    TrackPreselectorView,
    GNNTrackView,
    collate_fn,
)


def get_dataset(config, dataset_role: str = "train_path"):
    """
    Create a Dataset instance and corresponding collate function based on
    configuration.

    Parameters
    ----------
    config : object
        Configuration object. Expected fields:
            config.data.dataset_type : str
                One of {"production", "preselector"}.
            config.data.train_path : str
            config.data.val_path : str
            config.data.test_path : str

    dataset_role : str
        One of {"train_path", "val_path", "test_path"} indicating which
        dataset split to load.

    Returns
    -------
    dataset : torch.utils.data.Dataset
        Dataset instance corresponding to the requested view.

    collate : callable
        Collate function to be used with the DataLoader.
    """

    dataset_type = config.data.dataset_type

    if dataset_type == "dummy":
        raise ValueError(f"Dataset type no longer supported: {dataset_type}")

    # ------------------------------------------------------------------
    # Resolve path
    # ------------------------------------------------------------------
    if dataset_role == "train_path":
        path = config.data.train_path
    elif dataset_role == "val_path":
        path = config.data.val_path
    elif dataset_role == "test_path":
        path = config.data.test_path
    else:
        raise ValueError(f"Unknown dataset_role: {dataset_role}")

    # ------------------------------------------------------------------
    # Load base dataset (always the same)
    # ------------------------------------------------------------------
    base_dataset = TrackDatasetFromFile(path)

    # ------------------------------------------------------------------
    # Select view
    # ------------------------------------------------------------------
    if dataset_type == "production":
        dataset = TrackFullDNNView(base_dataset)
    
    elif dataset_type == "original":
        dataset = OriginalDataTrackDNNView(base_dataset)
    
    elif dataset_type == "preselector":
        dataset = TrackPreselectorView(base_dataset)
    
    elif dataset_type == "gnn":
        dataset = GNNTrackView(base_dataset)

    elif dataset_type == "dummy":
        raise ValueError(f"Dataset type no longer supported: {dataset_type}")

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # ------------------------------------------------------------------
    # Single collate function for all views
    # ------------------------------------------------------------------
    return dataset, collate_fn
