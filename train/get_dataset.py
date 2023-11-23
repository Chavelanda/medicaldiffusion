from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, DEFAULTDataset, SKULLBREAKDataset, SKULLBREAKDatasetTriplet
from torch.utils.data import WeightedRandomSampler

DATASET_CLASSES = {
    'MRNet': (MRNetDataset, {'train': {'task': None, 'plane': None, 'split': 'train'}, 'val': {'task': None, 'plane': None, 'split': 'valid'}}),
    'BRATS': (BRATSDataset, {'train': {'imgtype': None, 'train': True, 'severity': None, 'resize': None}, 'val': {'imgtype': None, 'train': False, 'severity': None, 'resize': None}}),
    'ADNI': (ADNIDataset, {'train': {'augmentation': True}, 'val': {'augmentation': False}}),
    'DUKE': (DUKEDataset, {'train': {}, 'val': {}}),
    'LIDC': (LIDCDataset, {'train': {'augmentation': True}, 'val': {'augmentation': False}}),
    'SKULL-BREAK': (SKULLBREAKDataset, {'train': {}, 'val': {}}),
    'SKULL-BREAK-TRIPLET': (SKULLBREAKDatasetTriplet, {'train': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}, 'val': {'resize_d': 1, 'resize_h': 1, 'resize_w': 1}}),
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

# def get_dataset(cfg):
#     if cfg.dataset.name == 'MRNet':
#         train_dataset = MRNetDataset(
#             root_dir=cfg.dataset.root_dir, task=cfg.dataset.task, plane=cfg.dataset.plane, split='train')
#         val_dataset = MRNetDataset(root_dir=cfg.dataset.root_dir,
#                                    task=cfg.dataset.task, plane=cfg.dataset.plane, split='valid')
#         sampler = WeightedRandomSampler(
#             weights=train_dataset.sample_weight, num_samples=len(train_dataset.sample_weight))
#         return train_dataset, val_dataset, sampler
#     if cfg.dataset.name == 'BRATS':
#         train_dataset = BRATSDataset(
#             root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=cfg.dataset.severity, resize=cfg.dataset.resize)
#         val_dataset = BRATSDataset(
#             root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=cfg.dataset.severity, resize=cfg.dataset.resize)
#         sampler = None
#         return train_dataset, val_dataset, sampler
#     if cfg.dataset.name == 'ADNI':
#         train_dataset = ADNIDataset(
#             root_dir=cfg.dataset.root_dir, augmentation=True)
#         val_dataset = ADNIDataset(
#             root_dir=cfg.dataset.root_dir, augmentation=True)
#         sampler = None
#         return train_dataset, val_dataset, sampler
#     if cfg.dataset.name == 'DUKE':
#         train_dataset = DUKEDataset(
#             root_dir=cfg.dataset.root_dir)
#         val_dataset = DUKEDataset(
#             root_dir=cfg.dataset.root_dir)
#         sampler = None
#         return train_dataset, val_dataset, sampler
#     if cfg.dataset.name == 'LIDC':
#         train_dataset = LIDCDataset(
#             root_dir=cfg.dataset.root_dir, augmentation=True)
#         val_dataset = LIDCDataset(
#             root_dir=cfg.dataset.root_dir, augmentation=True)
#         sampler = None
#         return train_dataset, val_dataset, sampler
#     if cfg.dataset.name == 'SKULL-BREAK':
#         train_dataset = SKULLBREAKDataset(
#             root_dir=cfg.dataset.root_dir)
#         val_dataset = SKULLBREAKDataset(
#             root_dir=cfg.dataset.root_dir)
#         sampler = None
#         return train_dataset, val_dataset, sampler
#     if cfg.dataset.name == 'DEFAULT':
#         train_dataset = DEFAULTDataset(
#             root_dir=cfg.dataset.root_dir)
#         val_dataset = DEFAULTDataset(
#             root_dir=cfg.dataset.root_dir)
#         sampler = None
#         return train_dataset, val_dataset, sampler
#     raise ValueError(f'{cfg.dataset.name} Dataset is not available')
