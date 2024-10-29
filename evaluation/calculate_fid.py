import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance

from dataset.get_dataset import get_dataset
from medicalnet import resnet_gap

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    print('Calculating FID with the following config:\n{}'.format(OmegaConf.to_yaml(cfg)))

    wandb.init(project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, name=cfg.model.run_name)

    wandb.config.update(OmegaConf.to_container(cfg.dataset))
    wandb.config.update(OmegaConf.to_container(cfg.model))

    dataset_real, dataset_gen, _ = get_dataset(cfg)
    dl_real = DataLoader(dataset_real, batch_size=cfg.model.batch_size, shuffle=True, num_workers=cfg.model.num_workers)
    dl_gen = DataLoader(dataset_gen, batch_size=cfg.model.batch_size, shuffle=True, num_workers=cfg.model.num_workers)

    model_params = {
        'sample_input_D': dataset_real.d,
        'sample_input_H': dataset_real.h,
        'sample_input_W': dataset_real.w,
        'num_seg_classes': 2, # Useless
        'no_cuda': not cfg.model.cuda
    }

    # the model outputs feature with shape (bs, 2048)
    model = resnet_gap(cfg.model.resnet, pretrain_path='medicalnet/pretrain/', **model_params)

    device = 'cuda' if cfg.model.cuda else 'cpu'

    model.to(device)

    fid = FrechetInceptionDistance(feature=model, normalize=True, input_img_size=(cfg.dataset.image_channels, dataset_real.d, dataset_real.h, dataset_real.w))
    
    # fid.set_dtype(torch.float64)
    fid.reset()

    for batch in tqdm(dl_real):
        # Update FID with real features
        batch = batch['data'].to(device)#.to(torch.float64)
        fid.update(batch, real=True)

    for batch in tqdm(dl_gen):
        # Update FID with gen features
        batch = batch['data'].to(device)#.to(torch.float64)
        fid.update(batch, real=False)

    # Compute FID
    fid_score = fid.compute()
    
    wandb.log({'fid': fid_score})
    print(f'FID score: {fid_score}')

if __name__ == '__main__':
    run()