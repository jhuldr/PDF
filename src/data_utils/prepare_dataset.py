from torch.utils.data import DataLoader

from data_utils.extend import ExtendedDataset
from data_utils.split import split_indices
from datasets.brain_lumiere import BrainLUMIEREMaskDataset, BrainLUMIEREMaskSubset
from datasets.brain_ms import BrainMSMaskDataset, BrainMSMaskSubset
from datasets.brain_ucsf import BrainUCSFMaskDataset, BrainUCSFMaskSubset
from utils.attribute_hashmap import AttributeHashmap


def prepare_dataset(config: AttributeHashmap, transforms_list=[None, None, None]):
    """Prepare longitudinal image+mask pairs for PDF training/evaluation."""

    if config.dataset_name == 'brain_lumiere_growth':
        dataset = BrainLUMIEREMaskDataset(
            image_folder='LUMIERE_images_axial_growth_tumor1200px_256x256/',
            mask_folder='LUMIERE_masks_axial_growth_tumor1200px_256x256/',
            target_dim=config.target_dim,
        )
        Subset = BrainLUMIEREMaskSubset

    elif config.dataset_name == 'brain_ucsf_growth':
        dataset = BrainUCSFMaskDataset(
            image_folder='brain_UCSF_images_axial_growth_whole_tumor1200px_256x256/',
            mask_folder='brain_UCSF_masks_axial_growth_whole_tumor1200px_256x256/',
            target_dim=config.target_dim,
        )
        Subset = BrainUCSFMaskSubset

    elif config.dataset_name == 'brain_ms_growth':
        dataset = BrainMSMaskDataset(
            image_folder='brain_MS_images_256x256/',
            mask_folder='brain_MS_masks_256x256/',
            target_dim=config.target_dim,
        )
        Subset = BrainMSMaskSubset

    else:
        raise ValueError(f'Unsupported PDF dataset: {config.dataset_name}')

    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple(c / sum(ratios) for c in ratios)
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = split_indices(
        indices=indices,
        splits=ratios,
        random_seed=config.random_seed,
    )

    if len(transforms_list) == 4:
        transforms_train, transforms_val, transforms_test, transforms_aug = transforms_list
    else:
        transforms_train, transforms_val, transforms_test = transforms_list
        transforms_aug = None

    train_set = Subset(
        main_dataset=dataset,
        subset_indices=train_indices,
        return_format='one_pair',
        transforms=transforms_train,
        transforms_aug=transforms_aug,
    )
    val_set = Subset(
        main_dataset=dataset,
        subset_indices=val_indices,
        return_format='all_pairs',
        transforms=transforms_val,
    )
    test_set = Subset(
        main_dataset=dataset,
        subset_indices=test_indices,
        return_format='one_pair',
        transforms=transforms_test,
    )

    min_sample_per_epoch = config.get('max_training_samples', 5)
    train_set = ExtendedDataset(dataset=train_set, desired_len=max(len(train_set), min_sample_per_epoch))

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader, test_loader, dataset.num_image_channel(), dataset.max_t
