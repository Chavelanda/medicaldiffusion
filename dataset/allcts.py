import os
import glob
import json

import torch
from torch.utils.data import Dataset
import torchio as tio

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nrrd


class AllCTsDataset(Dataset):
    def __init__(self, root_dir='data/AllCTs_nrrd_global', split='train', augmentation=False,
                 resize_d=1, resize_h=1, resize_w=1):
        
        assert split in ['all', 'train', 'val', 'test'], 'Invalid split: {}'.format(split)

        self.root_dir = root_dir

        # Read the CSV file as a DataFrame
        self.df = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
        self.df['name'] = self.df['name'].astype(str)

        # Take only the required split
        if split != 'all':
            self.df = self.df[self.df['split'] == split]

        # Read one 3d image and define sizes
        img, _ = nrrd.read(f'{self.root_dir}/{self.df["name"].iloc[0]}.nrrd')
        d, h, w = img.shape
      
        self.resize_d = resize_d
        self.resize_h = resize_h
        self.resize_w = resize_w

        # Update sizes based on resize
        self.d, self.h, self.w, = d//self.resize_d, h//self.resize_h, w//self.resize_w

        # Resize transform
        self.resize = tio.Resample((self.resize_d, self.resize_h, self.resize_w))

        # Augmentation transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.df['name'].iloc[index] + '.nrrd')
        img, _ = nrrd.read(path)

        img = torch.from_numpy(img)

        #  min-max normalized to the range between -1 and 1
        img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

        img = img.unsqueeze(0).float()
        img = self.resize(img)
       
        return {'data': img}
    
    def show_item(self, img, vmin=0, vmax=1):
        img = np.rot90(img, k=1, axes=(0, 2))

        # Get the slice index
        middle_slice = img.shape[0] // 2

        # Plot the middle slice from different perspectives
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Axial view
        axs[0].imshow(img[middle_slice, :, :], vmin=vmin, vmax=vmax, cmap='gray')
        axs[0].set_title('Axial View')

        # Sagittal view
        axs[1].imshow(img[:, middle_slice, :], vmin=vmin, vmax=vmax, cmap='gray')
        axs[1].set_title('Sagittal View')

        # Coronal view
        axs[2].imshow(img[:, :, middle_slice], vmin=vmin, vmax=vmax, cmap='gray')
        axs[2].set_title('Coronal View')

        # Show the plot
        plt.show()

    def get_named_item(self, item_name, vmin=0, vmax=1, path=None, show=True):
        if path is None:
            path = os.path.join(self.root_dir, item_name + '.nrrd')
        else:
            path = os.path.join(path, item_name + '.nrrd')
        
        img, _ = nrrd.read(path)

        if show:
            self.show_item(img, vmin, vmax)
        else:
            return img
        
    def get_named_item_from_dataset(self, item_name):
        index = self.df[self.df['name'] == item_name].index[0]
        
        return self.__getitem__(index)

    def save_to_nrrd(self, item_name, item, save_path=None):
        # Transform the item to numpy array
        item = item.numpy()

        # Remove channel dimension if present
        if len(item.shape) == 4:
            item = item.squeeze(0)

        #  min-max normalized to the range between 0 and 1
        item = (item - item.min()) / (item.max() - item.min())
        
        if save_path is None:
            save_path = os.path.join(self.root_dir, item_name + '.nrrd')
        
        nrrd.write(save_path, item)

    def show_image_grid(self, images, axis=2, slices=(0,20,40,60,80), vmax=1, vmin=-1, names=None):
        if isinstance(images[0], torch.Tensor):
            images = [image.detach().cpu().numpy() for image in images]
        
        # Remove channel and batch dimensions if present and rotate images
        images = [np.rot90(np.squeeze(image), k=1, axes=(0, 2)) for image in images]

        # Select slices and axis and , if present
        sliced_images = [np.swapaxes(np.take(image, slices, axis=axis), 0, axis) for image in images]
        
        # Display images
        img_w = 2 * len(slices)
        img_h = 2 * len(images)
        
        fig, axs = plt.subplots(len(images), len(slices), figsize=(img_w, img_h))
        
        for i, image in enumerate(sliced_images):
            
            for j, slice in enumerate(image):
                if len(image) == 1:
                    ax = axs[i]
                else:
                    ax = axs[i, j]
                ax.imshow(slice, vmin=vmin, vmax=vmax, cmap='gray')
                
                ax.xaxis.set_visible(False)
                plt.setp(ax.spines.values(), visible=False)
                ax.tick_params(left=False, labelleft=False)
                
                if i == 0:                
                    ax.set_title(f'Slice {slices[j]}')

        if names is not None:
            for i, name in enumerate(names):
                if len(slice) == 1:
                    ax = axs[i]
                else:
                    ax = axs[i, 0]
                ax.set_ylabel(name, rotation=90, size='medium')
        
        fig.tight_layout()    
        plt.show()



def process_mesh_folder_and_save_voxel_grid(input_folder, output_folder, grid_dim=512, file_start=None):
    start = True if file_start is None else False

    for idx, folder_name in enumerate(os.listdir(input_folder)):
        folder_path = os.path.join(input_folder, folder_name)
        print(f"Processing {folder_path}")
        if start and os.path.isdir(folder_path):
            # Assuming the folder contains two files "{name}_C.stl" and "{name}_M.stl"
            file_c_path = os.path.join(folder_path, f"{folder_name}_C.stl")
            file_m_path = os.path.join(folder_path, f"{folder_name}_M.stl")
            
            if os.path.exists(file_c_path) and os.path.exists(file_m_path):
                print(f"Found {file_c_path} and {file_m_path}")
                # Load meshes
                mesh_c = o3d.io.read_triangle_mesh(file_c_path)
                mesh_m = o3d.io.read_triangle_mesh(file_m_path)
                
                # Merge meshes
                merged_mesh = mesh_c + mesh_m

                # Create a scene and add the triangle mesh
                scene_mesh = o3d.t.geometry.TriangleMesh.from_legacy(merged_mesh)
                scene = o3d.t.geometry.RaycastingScene()
                _ = scene.add_triangles(scene_mesh)  # we do not need the geometry ID for mesh

                min_bound = scene_mesh.vertex.positions.min(0).numpy()
                max_bound = scene_mesh.vertex.positions.max(0).numpy()

                xyz_range = np.linspace(min_bound, max_bound, num=grid_dim)
                query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)

                # occupancy is a [grid_dim,grid_dim,grid_dim] array
                occupancy = scene.compute_occupancy(query_points)
                
                print('Writing to nrrd')
                save_path = os.path.join(output_folder, f"{folder_name}.nrrd")
                nrrd.write(save_path, occupancy.numpy())
                
                # pytorch_occupancy = torch.utils.dlpack.from_dlpack(occupancy.to_dlpack()).bool()

                # print(pytorch_occupancy.shape, pytorch_occupancy.dtype, pytorch_occupancy.device)
                
                # # Save voxel grid
                # save_path = os.path.join(folder_path, f"{folder_name}.pt")
                # torch.save(pytorch_occupancy, save_path)
                # print(f"Voxel grid saved for {folder_name} at {save_path}")
        
        if folder_name == file_start:
            print(f'\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!\nLast unprocessed folder is {folder_name}. Next folder processed will create the {idx+2}th file\n!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n')
            start = True
   
def rename_files(folder):
    # Get all .nrrd files in the folder
    files = glob.glob(os.path.join(
            folder, './**/*.nrrd'), recursive=True)

    for file in files:
        # Split the file path into folder, name and extension
        folder, name = os.path.split(file)
        base, ext = os.path.splitext(name)

        # Split the base into letters and number
        letters = base.rstrip('0123456789')
        number = base[len(letters):]

        # Rename the file
        new_name = '{}{:03}{}'.format(letters, int(number), ext)
        os.rename(file, os.path.join(folder, new_name))

if __name__ == '__main__':
    dataset = AllCTsDataset(root_dir='data/allcts-global-gen-01', split='all')
    print(len(dataset))
    item = dataset.__getitem__(0)

    dataset.show_item(item['data'].squeeze(0).numpy())
    