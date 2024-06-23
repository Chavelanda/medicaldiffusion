import os
import csv
from dataset.allcts import AllCTsDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vq_gan_3d.model import VQGAN

def run():
    ds = AllCTsDataset('data/allcts-global-128', split='test', metadata_name=f'metadata.csv', binarize=True)
    
    batch_size = 1
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=10)

    model_name = 'vqgan-06'

    model = VQGAN.load_from_checkpoint(f'checkpoints/vq_gan_3d/AllCTs/{model_name}/best_val.ckpt').cuda()
    data_path = f'data/{model_name}/'
    metadata_path = os.path.join(data_path, 'metadata.csv')

    # Create metadata csv if not existing
    with open(metadata_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # If the file is empty, write the header
        if os.stat(metadata_path).st_size == 0:
            writer.writerow(['name', 'split', 'quality'])

    # Reconstructing test set
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl)):
            name = ds.df.iloc[i, :]['name'] + '-recon'
            
            data = batch['data'].cuda()
            cond = batch['cond']
            batch = {'data': data}
            recon = model.test_step(batch, 0).cpu()
            
            AllCTsDataset.save(name, recon, f'data/{model_name}/')

            class_name = ds.get_class_name_from_cond(cond)[0]

            # Append metadata to csv
            with open(metadata_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, 'test', class_name])


if __name__ == '__main__':
    run()