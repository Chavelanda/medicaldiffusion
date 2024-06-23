import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from train.get_dataset import get_dataset
from medicalnet import resnet_gap
from evaluation.fid import compute_stats_from_model


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    with open_dict(cfg):
        cfg.model.name = f'{cfg.model.name}-{cfg.dataset.metadata_name[-5]}'
    print('Calculating FID stats with the following config:\n{}'.format(OmegaConf.to_yaml(cfg)))

    dataset, *_ = get_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=cfg.model.num_workers)

    stats_dir = cfg.model.stats_dir
    name = cfg.model.name

    model_params = {
        'sample_input_D': dataset.d,
        'sample_input_H': dataset.h,
        'sample_input_W': dataset.w,
        'num_seg_classes': 2, # Useless
        'no_cuda': not cfg.model.cuda
    }

    model = resnet_gap(cfg.model.resnet, pretrain_path='medicalnet/pretrain/', **model_params)

    device = 'cuda' if cfg.model.cuda else 'cpu'

    print('Computing stats...')
    mu, sigma = compute_stats_from_model(model, dataloader, device, name, stats_dir, save=True)

    print('Stats shapes:\nmu {}\nsigma {}'.format(mu.shape, sigma.shape))
    print('Done! FID stats saved in {}.'.format(stats_dir))


if __name__ == '__main__':
    run()