import os
import glob
import json

import torch
from torch.utils.data import Dataset
import torchio as tio

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import nrrd

class AllCTsDataset(Dataset):
    def __init__(self, root_dir='data/AllCTs_nrrd', split='train', augmentation=False,
                 resize_d=1, resize_h=1, resize_w=1):
        
        assert split in ['train', 'val', 'test'], 'Invalid split: {}'.format(split)
        
        self.cts = []

        self.root_dir = root_dir

        # Take json file containing splits
        with open(os.path.join(root_dir, 'splits.json'), 'r') as f:
            splits = json.load(f)
        
        # Take only the required split
        for quality in splits:
            self.cts += quality[split]

        # Read one 3d image and define sizes
        img, _ = nrrd.read(f'{self.root_dir}/{self.cts[0]}.nrrd')
        d, h, w = img.shape
        # Show one slice
        # plt.imshow(img[d//2,:,:], cmap='gray')
        # plt.show()
        self.resize_d = resize_d
        self.resize_h = resize_h
        self.resize_w = resize_w

        # Update sizes based on resize
        self.d, self.h, self.w, = d//self.resize_d, h//self.resize_h, w//self.resize_w

        # Resize transform
        self.resize = tio.Resample((self.resize_d, self.resize_h, self.resize_w))

        # Augmentation transform

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.cts[index] + '.nrrd')
        img, _ = nrrd.read(path)

        img = torch.from_numpy(img)
        img = img.unsqueeze(0).float()
        img = self.resize(img)
       
        return {'data': img}

def process_folders_and_save_voxel_grid(input_folder, output_folder, grid_dim=512, file_start=None):
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

    # mesh = trimesh.load('data/AllCTs/CT2/CT2_C.stl')
    # mesh.visual.face_colors = [200, 200, 250, 100]
    # mesh_m = trimesh.load('data/AllCTs/CT2/CT2_M.stl')
    # print(mesh.extents)
    # voxelized = mesh.voxelized(0.66).fill()
    # voxelized_m = mesh.voxelized(0.66).fill()
    # print(voxelized.shape)
    # # Convert voxelized mesh to a 3D numpy array
    # voxel_array = voxelized.matrix
    # print(type(voxel_array))

    # # Select a slice to display
    # slice_index = voxel_array.shape[2] // 2  # Select the middle slice
    # slice = voxel_array[:, :, slice_index]
    
    # plt.imshow(slice, cmap='gray')
    # plt.show()

    # combined = trimesh.util.concatenate([mesh, mesh_m])
    # s = trimesh.Scene(combined)
    # s.show()

    # mesh = o3d.io.read_triangle_mesh('data/AllCTs/CT2/CT2_C.stl')
    # # print(mesh)
    # # mesh.compute_vertex_normals()
    # # mesh.paint_uniform_color([1, 0.706, 0])
    # # o3d.visualization.draw_geometries([mesh])
    
    # rename_files('data/AllCTs_nrrd')

    # Example usage:
    # input_folder_path = "data/AllCTs/"
    # output_folder_path = "data/AllCTs_nrrd/"
    # file_start = 'CTS242'
    # process_folders_and_save_voxel_grid(input_folder_path, output_folder_path, file_start=file_start)

    dataset = AllCTsDataset(split='val')
    print(dataset.cts[0])
    print(len(dataset))
    input = dataset[1]
    
    plt.imshow(input['data'][0, :,512//2,:], cmap='gray')
    plt.show()