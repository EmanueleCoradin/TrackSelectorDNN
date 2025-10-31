from dataset import TrackDatasetFromFile, collate_fn as prod_collate
from dummy_dataset import DummyTrackDataset, collate_fn as dummy_collate

def get_dataset(config):
    """
    Returns a Dataset instance and corresponding collate_fn.
    Selection depends on config["dataset_type"].
    """
    dataset_type = config["dataset_type"]

    if dataset_type == "dummy":
        dataset = DummyTrackDataset(
            n_tracks=config["n_tracks"],
            hit_input_dim=config["hit_input_dim"],
            track_feat_dim=config["track_feat_dim"],
            max_hits=config["max_hits"], 
        )
        collate = dummy_collate

    elif dataset_type == "production":
        dataset = TrackDatasetFromFile(config["train_path"])
        collate = prod_collate

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return dataset, collate