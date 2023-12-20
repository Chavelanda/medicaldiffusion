from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, DEFAULTDataset, SKULLBREAKDataset, SKULLBREAKDatasetTriplet, AllCTsDataset
from torch.utils.data import WeightedRandomSampler

DATASET_CLASSES = {
    'MRNet': (MRNetDataset, {'train': {'task': None, 'plane': None, 'split': 'train'}, 'val': {'task': None, 'plane': None, 'split': 'valid'}}),
    'BRATS': (BRATSDataset, {'train': {'imgtype': None, 'train': True, 'severity': None, 'resize': None}, 'val': {'imgtype': None, 'train': False, 'severity': None, 'resize': None}}),
    'ADNI': (ADNIDataset, {'train': {'augmentation': True}, 'val': {'augmentation': False}}),
    'DUKE': (DUKEDataset, {'train': {}, 'val': {}}),
    'LIDC': (LIDCDataset, {'train': {'augmentation': True}, 'val': {'augmentation': False}}),
    'SKULL-BREAK': (SKULLBREAKDataset, {'train': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}, 'val': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}}),
    'SKULL-BREAK-TRIPLET': (SKULLBREAKDatasetTriplet, {'train': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}, 'val': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}}),
    'AllCTs': (AllCTsDataset, {'train': {'split': 'all', 'resize_d': 1, 'resize_h': 1, 'resize_w': 1}, 'val': {'split': 'val', 'resize_d': 1, 'resize_h': 1, 'resize_w': 1}}),
    'DEFAULT': (DEFAULTDataset, {'train': {}, 'val': {}})
}

def get_dataset(cfg):
    DatasetClass, dataset_params = DATASET_CLASSES[cfg.dataset.name]
    train_params = dataset_params['train'].copy()
    val_params = dataset_params['val'].copy()
    train_params['root_dir'] = cfg.dataset.root_dir
    val_params['root_dir'] = cfg.dataset.root_dir
    for key in train_params:
        if key in cfg.dataset:
            train_params[key] = cfg.dataset[key]
    for key in val_params:
        if key in cfg.dataset:
            val_params[key] = cfg.dataset[key]
    train_dataset = DatasetClass(**train_params)
    val_dataset = DatasetClass(**val_params)
    if cfg.dataset.name == 'MRNet':
        sampler = WeightedRandomSampler(weights=train_dataset.sample_weight, num_samples=len(train_dataset.sample_weight))
    else:
        sampler = None
    return train_dataset, val_dataset, sampler