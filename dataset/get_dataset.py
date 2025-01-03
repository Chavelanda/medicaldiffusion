from dataset import MRNetDataset, MRNetDatasetMSSSIM, MRNetDatasetSS, DEFAULTDataset, SKULLBREAKDataset, SKULLBREAKDatasetTriplet, AllCTsDataset, AllCTsDatasetSS, AllCts_MSSSIM, AllCTsDatasetUpsampling
from torch.utils.data import WeightedRandomSampler

DATASET_CLASSES = {
    'MRNet': (MRNetDataset, {'train': {'split': 'train', 'conditioned': True, 'metadata_name': 'metadata.csv', 'augment': False}, 'val': {'split': 'val', 'conditioned': True, 'metadata_name': 'metadata.csv', 'augment': False}}),
    'MRNetMSSSIM': (MRNetDatasetMSSSIM, {'train': {'split': 'train', 'metadata_name': 'metadata.csv', 'samples': 1000}, 'val': {'split': 'val', 'metadata_name': 'metadata.csv', 'samples': 1000}}),
    'MRNetSS': (MRNetDatasetSS, {'train': {'split': 'train', 'metadata_name': 'metadata.csv', 'recon_root_dir': None, 'recon_metadata_name': 'metadata.csv'}, 'val': {'split': 'val', 'metadata_name': 'metadata.csv', 'recon_root_dir': None, 'recon_metadata_name': 'metadata.csv'}}),
    'SKULL-BREAK': (SKULLBREAKDataset, {'train': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}, 'val': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}}),
    'SKULL-BREAK-TRIPLET': (SKULLBREAKDatasetTriplet, {'train': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}, 'val': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}}),
    'AllCTs': (AllCTsDataset, {'train': {'split': 'train-val', 'qs': None, 'resample': 1, 'rescale': True, 'conditioned': True, 'binarize': False, 'metadata_name': 'metadata.csv'}, 'val': {'split': 'test', 'qs': None, 'resample': 1, 'rescale': True, 'conditioned': True, 'binarize': False, 'metadata_name': 'metadata.csv'}}),
    'AllCTsSS': (AllCTsDatasetSS, {'train': {'split': 'train-val', 'resample': 1, 'rescale': True, 'binarize': False, 'metadata_name': 'metadata.csv', 'recon_root_dir': None, 'recon_metadata_name': 'metadata.csv'}, 'val': {'split': 'test', 'resample': 1, 'rescale': True, 'binarize': False, 'metadata_name': 'metadata.csv', 'recon_root_dir': None, 'recon_metadata_name': 'metadata.csv'}}),
    'allcts-msssim': (AllCts_MSSSIM, {'train': {'split': 'train-val', 'samples': 1000, 'resample': 1, 'rescale': True, 'binarize': False, 'metadata_name': 'metadata.csv'}, 'val': {'split': 'test', 'resample': 1, 'rescale': True, 'binarize': False, 'metadata_name': 'metadata.csv'}}),
    'AllCTs-Upsampling': (AllCTsDatasetUpsampling, {'train': {'split': 'train-val', 'qs': None, 'resample': 1, 'rescale': True, 'conditioned': True, 'binarize': False, 'metadata_name': 'metadata.csv'}, 'val': {'split': 'test', 'qs': None, 'resample': 1, 'rescale': True, 'conditioned': True, 'binarize': False, 'metadata_name': 'metadata.csv'}}),
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