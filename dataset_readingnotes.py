

import pytorch_lightning as pl
import torch
import wandb
import yaml
from omegaconf import OmegaConf
from pycg import exp
from scube.data.base import DatasetSpec as DS
from scube.modules.gsm_modules.hparams import hparams_handler as gsm_hparams_handler
from scube.modules.autoencoding.hparams import hparams_handler as autoencoding_hparams_handler
from scube.modules.diffusionmodules.hparams import hparams_handler as diffusion_hparams_handler
import torchvision
import imageio.v3 as imageio
from matplotlib import pyplot as plt
import os

import scube.data as dataset

VIS_FOLDER = "/root/SCube/outputs/vis_waymo_wds"

def get_dataset_spec(model_name, hparams):
    if model_name == 'autoencoder':
        all_specs = [DS.SHAPE_NAME, DS.INPUT_PC,
                     DS.GT_DENSE_PC, DS.GT_GEOMETRY]
        hparams = autoencoding_hparams_handler(hparams)
        if hparams.use_input_normal:
            all_specs.append(DS.TARGET_NORMAL)
            all_specs.append(DS.GT_DENSE_NORMAL)
        if hparams.use_input_semantic or hparams.with_semantic_branch:
            all_specs.append(DS.GT_SEMANTIC)
        if hparams.use_input_intensity:
            all_specs.append(DS.INPUT_INTENSITY)

    elif model_name == 'diffusion':
        all_specs = get_dataset_spec('autoencoder', hparams)
        hparams = diffusion_hparams_handler(hparams)
        # further add new specs
        if hparams.use_semantic_cond:
            all_specs.append(DS.LATENT_SEMANTIC)
        if hparams.use_single_scan_concat_cond:
            all_specs.append(DS.SINGLE_SCAN_CROP)
            all_specs.append(DS.SINGLE_SCAN)
        if hparams.use_class_cond:
            all_specs.append(DS.CLASS)
        if hparams.use_text_cond:
            all_specs.append(DS.TEXT_EMBEDDING)
            all_specs.append(DS.TEXT_EMBEDDING_MASK)
        if hparams.use_image_w_depth_cond or hparams.use_image_lss_cond:
            all_specs.append(DS.IMAGES_INPUT)
            all_specs.append(DS.IMAGES_INPUT_DEPTH)
            all_specs.append(DS.IMAGES_INPUT_MASK)
        if hparams.use_map_3d_cond:
            all_specs.append(DS.MAPS_3D)
        if hparams.use_box_3d_cond:
            all_specs.append(DS.BOXES_3D)
        if hparams.use_micro_cond:
            all_specs.append(DS.MICRO)
        return all_specs

    elif model_name == "gsm":
        all_specs = [DS.SHAPE_NAME, DS.INPUT_PC, DS.GT_SEMANTIC]
 
        all_specs.append(DS.IMAGES_INPUT)
        all_specs.append(DS.IMAGES_INPUT_MASK)
        all_specs.append(DS.IMAGES_INPUT_POSE)
        all_specs.append(DS.IMAGES_INPUT_INTRINSIC)

        all_specs.append(DS.IMAGES)
        all_specs.append(DS.IMAGES_MASK)
        all_specs.append(DS.IMAGES_POSE)
        all_specs.append(DS.IMAGES_INTRINSIC)

        hparams = gsm_hparams_handler(hparams)
        
        if hparams.use_sup_depth and hparams.sup_depth_type == 'rectified_metric3d_depth':
            all_specs.append(DS.IMAGES_DEPTH_MONO_EST_RECTIFIED)
        if hparams.use_sup_depth and hparams.sup_depth_type == 'lidar_depth':
            all_specs.append(DS.IMAGES_DEPTH_LIDAR_PROJECT)
        if hparams.use_sup_depth and hparams.sup_depth_type == 'depth_anything_v2_depth_inv':
            all_specs.append(DS.IMAGES_DEPTH_ANYTHING_V2_DEPTH_INV)
        if hparams.use_sup_depth and hparams.sup_depth_type == 'voxel_depth':
            pass # voxel depth is generated on the fly

    return all_specs

def visualize_images(image_tensor):
    # image_tensor: [N, H, W, 3]
    grid_img = torchvision.utils.make_grid(image_tensor.permute(0, 3, 1, 2), nrow=image_tensor.shape[0], padding=2)
    return grid_img

def visualize_depth(depth_tensor):
    # depth_tensor: [N, H, W, 1]
    depth_img = []
    for ii in range(depth_tensor.shape[0]):
        downsample_factor = 4  # Adjust this factor as needed
        depth_map = depth_tensor[ii, :, :, :]

        depth_downsampled = depth_tensor[ii, ::downsample_factor, ::downsample_factor, :]
        max_95 = torch.quantile(depth_downsampled, 0.95)
        min_5 = torch.quantile(depth_downsampled, 0.05)

        depth_clipped = depth_map.clamp(min_5, max_95)
        depth_clipped = (depth_clipped - depth_clipped.min()) / (depth_clipped.max() - depth_clipped.min())
        # get a warm colormap
        colormap = plt.cm.get_cmap('magma')
        depth_img_colored = colormap(depth_clipped)

        depth_img.append(depth_img_colored[:, :, :, :3].squeeze())

    depth_img = torch.tensor(depth_img)
    grid_img = torchvision.utils.make_grid(depth_img.permute(0, 3, 1, 2), nrow=depth_img.shape[0], padding=2)
    return grid_img

def visualize_mask(mask_tensor):
    # mask_tensor: [N, H, W, n_masks]
    # use torchvision and grid to visualize them
    mask_grid_img = []
    for ii in range(mask_tensor.shape[-1]):
        mask_img = mask_tensor[:, :, :, ii:ii+1]
        # mask_tensor is now [n_views, H, W, 1]
        grid_img = torchvision.utils.make_grid(mask_img.permute(0, 3, 1, 2), nrow=mask_img.shape[0], padding=2)
        mask_grid_img.append(grid_img)
    return mask_grid_img

def print_sample_dict(sample_dict, depth=0, visualize=False,image_list=[]):
    prefix = '    ' * depth

    for key, value in sample_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{prefix}{key:<40}: type: {type(value).__name__:<10}, shape: {value.shape}")
        elif isinstance(value, str):
            print(f"{prefix}{key:<40}: type: {type(value).__name__:<10}, value: {value}")
        elif isinstance(value, dict):
            print(f"{prefix}{key:<40}: type: {type(value).__name__:<10}")
            print_sample_dict(value, depth=depth+1, visualize=visualize, image_list=image_list)
        else:
            print(f"{prefix}{key:<40}: type: {type(value).__name__:<10}")
    
        # visualize all together.    
        if visualize:
            # image, depth, semantic, mask. I want to visualize them..
            # I also want to visualize the point cloud, as well as the grid.
            if key == DS.IMAGES_INPUT:
                image_list.append(visualize_images(value))
            elif key == DS.IMAGES_INPUT_DEPTH:
                image_list.append(visualize_depth(value))
            elif key == DS.IMAGES_INPUT_MASK:
                image_list += visualize_mask(value) # return a list of image_grids.
            
            

if __name__ == '__main__':
    """""""""""""""""""""""""""""""""""""""""""""""
    [1] Parse and initialize program arguments
        these include: --debug, --profile, --gpus, --num_nodes, --resume, ...
        they will NOT be saved for a checkpoints.
    """""""""""""""""""""""""""""""""""""""""""""""
    program_parser = exp.argparse.ArgumentParser()
    program_parser = pl.Trainer.add_argparse_args(program_parser)
    program_args, other_args = program_parser.parse_known_args()

    model_parser = exp.ArgumentParserX(base_config_path='configs/default/param.yaml')
    model_args = model_parser.parse_args(other_args)
    # hyper_path = model_args.hyper
    # del model_args["hyper"]
    hparams = OmegaConf.to_container(model_args, resolve=True)
    hparams = OmegaConf.create(hparams)

    val_set = dataset.build_dataset(
        hparams.val_dataset, get_dataset_spec(hparams.model, hparams), hparams, hparams.val_kwargs
    )
    x = next(iter(val_set))

    image_list = []
    print_sample_dict(x, depth=0, visualize=True, image_list=image_list)

    grid_img = torch.stack(image_list, dim=0) # [N, 3, H, W]
    grid_img = torchvision.utils.make_grid(grid_img, nrow=1, padding=2) # [3, H * N, W]
    grid_img_uint8 = (grid_img.permute(1, 2, 0).numpy() * 255).astype('uint8')

    model_name = model_args.hyper.split('/')[-1].split('.')[0]

    os.makedirs(VIS_FOLDER, exist_ok=True)
    imageio.imwrite(os.path.join(VIS_FOLDER, f"{model_name}_example_frame.png"), grid_img_uint8)