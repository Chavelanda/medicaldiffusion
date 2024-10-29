import os
import glob

import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt
import nrrd
import mcubes

def show_item(img, vmin=0, vmax=1, name='foo'):
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
    plt.savefig(f'{name}.png')
    plt.show()

def show_image_grid(images, axis=2, slices=(0,20,40,60,80), vmax=1, vmin=-1, names=None):
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


# Process a folder containing meshes and save the voxel grid in the specified output folder
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
                     
        if folder_name == file_start:
            print(f'\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!\nLast unprocessed folder is {folder_name}. Next folder processed will create the {idx+2}th file\n!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n')
            start = True


# Rename files in a folder to have a 3 digit number
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


# Build mesh from voxel grid
def build_mesh(voxels, threshold=0.5, name='mesh', output_folder=None, spacing=[0.46262616, 0.3345859, 0.49944812]):
    # voxels = mcubes.smooth(voxels)
    vertices, triangles = mcubes.marching_cubes(voxels, threshold)

    # Scale the vertices according to the actual voxel spacing
    vertices = vertices * np.array(spacing)

    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices), triangles=o3d.utility.Vector3iVector(triangles))
    
    print(mesh)
    
    mesh.compute_vertex_normals()

    if output_folder is not None:
        output_folder = os.path.join(output_folder, name + '.stl')
        o3d.io.write_triangle_mesh(output_folder, mesh)
    else:
        o3d.visualization.draw_geometries([mesh])

def build_mesh_from_file(file_path, threshold=0.5, name='mesh', output_folder=None):
    voxels, _ = nrrd.read(file_path)

    build_mesh(voxels, threshold=threshold, name=name, output_folder=output_folder)

def build_mesh_from_folder(input_folder, output_folder, threshold=0.5):
    for file_name in os.listdir(input_folder):
        if not file_name.endswith('.nrrd'):
            continue
        print(f'Processing {file_name}')
        
        name = os.path.splitext(file_name)[0]
        file_path = os.path.join(input_folder, file_name)
        
        build_mesh_from_file(file_path, threshold=threshold, name=name, output_folder=output_folder)

if __name__ == '__main__':
    build_mesh_from_folder('data/mesh-test', 'data/mesh-test')
