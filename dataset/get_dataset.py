from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, DEFAULTDataset, SKULLBREAKDataset, SKULLBREAKDatasetTriplet, AllCTsDataset, AllCTsDatasetSS, AllCts_MSSSIM
from torch.utils.data import WeightedRandomSampler

DATASET_CLASSES = {
    'MRNet': (MRNetDataset, {'train': {'split': 'train', 'conditioned': True, 'metadata_name': 'metadata.csv'}, 'val': {'split': 'val', 'conditioned': True, 'metadata_name': 'metadata.csv'}}),
    'BRATS': (BRATSDataset, {'train': {'imgtype': None, 'train': True, 'severity': None, 'resize': None}, 'val': {'imgtype': None, 'train': False, 'severity': None, 'resize': None}}),
    'ADNI': (ADNIDataset, {'train': {'augmentation': True}, 'val': {'augmentation': False}}),
    'DUKE': (DUKEDataset, {'train': {}, 'val': {}}),
    'LIDC': (LIDCDataset, {'train': {'augmentation': True}, 'val': {'augmentation': False}}),
    'SKULL-BREAK': (SKULLBREAKDataset, {'train': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}, 'val': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}}),
    'SKULL-BREAK-TRIPLET': (SKULLBREAKDatasetTriplet, {'train': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}, 'val': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}}),
    'AllCTs': (AllCTsDataset, {'train': {'split': 'train-val', 'resize_d': 1, 'resize_h': 1, 'resize_w': 1, 'conditioned': True, 'binarize': False, 'metadata_name': 'metadata.csv'}, 'val': {'split': 'test', 'resize_d': 1, 'resize_h': 1, 'resize_w': 1, 'conditioned': True, 'binarize': False, 'metadata_name': 'metadata.csv'}}),
    'AllCTsSS': (AllCTsDatasetSS, {'train': {'split': 'train-val', 'resize_d': 1, 'resize_h': 1, 'resize_w': 1, 'binarize': False, 'metadata_name': 'metadata.csv'}, 'val': {'split': 'test', 'resize_d': 1, 'resize_h': 1, 'resize_w': 1, 'binarize': False, 'metadata_name': 'metadata.csv'}}),
    'allcts-msssim': (AllCts_MSSSIM, {'train': {'split': 'train-val', 'samples': 1000, 'resize_d': 1, 'resize_h': 1, 'resize_w': 1, 'binarize': False, 'metadata_name': 'metadata.csv'}, 'val': {'split': 'test', 'resize_d': 1, 'resize_h': 1, 'resize_w': 1, 'binarize': False, 'metadata_name': 'metadata.csv'}}),
    'DEFAULT': (DEFAULTDataset, {'train': {}, 'val': {}})
}

def get_dataset(cfg):
    DatasetClass, dataset_params = DATASET_CLASSES[cfg.dataset.name]
    train_params = dataset_params['train'].copy()
    val_params = dataset_params['val'].copy()
    train_params['root_dir'] = cfg.dataset.root_dir
    val_params['root_dir'] = cfg.dataset.val_root_dir
    # Setting train params
    for key in train_params:
        if key in cfg.dataset:
            train_params[key] = cfg.dataset[key]
    # Using train params also for validation
    for key in val_params:
        if key in cfg.dataset:
            val_params[key] = cfg.dataset[key]
    # Overwriting val params when specified
    for key in val_params:
        config_key = 'val_' + key
        if config_key in cfg.dataset:
            val_params[key] = cfg.dataset[config_key]
    print(f'Training parameters\n{train_params}')
    print(f'Validation parameters\n{val_params}')
    train_dataset = DatasetClass(**train_params)
    val_dataset = DatasetClass(**val_params)
    # if cfg.dataset.name == 'MRNet':
    #     sampler = WeightedRandomSampler(weights=train_dataset.sample_weight, num_samples=len(train_dataset.sample_weight))
    # else:
    sampler = None
    return train_dataset, val_dataset, sampler