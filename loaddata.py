from torch.utils.data import DataLoader
from ultralytics.data.dataset import YOLODataset

def create_validation_loader(img_folders, batch_size=1):
    """
    Create a validation DataLoader using multiple image folder paths
    
    Args:
        img_folders: List of paths to folders containing validation images
        batch_size: Batch size for validation
    
    Returns:
        DataLoader for combined validation data
    """
    # Class names from your dataset
    class_names = {
        0: 'Metal', 
        1: 'Plastic', 
        2: 'Glass', 
        3: 'Cardboard', 
        4: 'Paper', 
        5: 'Organic', 
        6: 'Wood', 
        7: 'e-Waste', 
        8: 'Rubble', 
        9: 'Fabric'
    }
    
    # Create datasets from each folder
    datasets = []
    for folder in img_folders:
        dataset = YOLODataset(
            img_path=folder,
            data={"names": class_names  , "channels": 3},
            task="detect",
            augment=False,
            cache=False,
            single_cls=False,
            rect=False,
            prefix='val'
        )
        datasets.append(dataset)
    
    # Combine datasets
    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset(datasets)
    
    # Create a DataLoader with the combined dataset
    valid_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=datasets[0].collate_fn,  # Use collate_fn from first dataset
        drop_last=False
    )
    
    return valid_loader


if __name__ == "__main__":
    
    
    # Example usage
    valid_loader = create_validation_loader(
        ["serbeco/images/val"], 
        batch_size=1)

