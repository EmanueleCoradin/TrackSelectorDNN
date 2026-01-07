from TrackSelectorDNN.data_manager.dataset import TrackDatasetFromFile, TrackPreselectorDatasetFromFile, collate_fn as prod_collate, preselector_collate_fn as pre_collate

from TrackSelectorDNN.data_manager.dummy_dataset import DummyTrackDataset, collate_fn as dummy_collate

def get_dataset(config, dataset_role="train_path"):
    """
    Returns a Dataset instance and corresponding collate_fn.
    Selection depends on config["dataset_type"].
    """
    dataset_type = config.data.dataset_type
    
    if dataset_type == "dummy":
        raise ValueError(f"Dataset type no longer supported: {dataset_type}")
    
    elif dataset_type == "production":
        collate = prod_collate
        if dataset_role == "train_path":
            dataset = TrackDatasetFromFile(config.data.train_path)
        elif dataset_role == "val_path":
            dataset = TrackDatasetFromFile(config.data.val_path)
        elif dataset_role == "test_path":
            dataset = TrackDatasetFromFile(config.data.test_path)
        else:
            raise ValueError(f"Unknown dataset_role: {dataset_role}")
    
    elif dataset_type == "preselector":
        collate = pre_collate
        if dataset_role == "train_path":
            dataset = TrackPreselectorDatasetFromFile(config.data.train_path)
        elif dataset_role == "val_path":
            dataset = TrackPreselectorDatasetFromFile(config.data.val_path)
        elif dataset_role == "test_path":
            dataset = TrackPreselectorDatasetFromFile(config.data.test_path)
        else:
            raise ValueError(f"Unknown dataset_role: {dataset_role}")
    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return dataset, collate