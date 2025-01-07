# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import importlib
import torch
import fvdb
import fvdb.nn
import pytorch_lightning as pl
import sys
import os
import polyscope as ps

from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

from omegaconf import OmegaConf
from fvdb import JaggedTensor, GridBatch
from fvdb.nn import VDBTensor
from pathlib import Path
from pycg import exp
from scube.utils import wandb_util
from loguru import logger 
from tqdm import tqdm
from scube.utils.common_util import batch2device, get_default_parser, create_model_from_args
from scube.utils.gaussian_util import save_splat_file_RGB
from scube.data.base import DatasetSpec as DS
from scube.utils import exp
from scube.utils.vis_util import visualize_structure_recon, visualize_normal_recon


fvdb.nn.SparseConv3d.backend = 'igemm_mode1'

def get_default_parser():
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser = pl.Trainer.add_argparse_args(default_parser)
    return default_parser

def create_model_from_args(config_path, ckpt_path, strict=True):
    model_yaml_path = Path(config_path)
    model_args = exp.parse_config_yaml(model_yaml_path)
    net_module = importlib.import_module("scube.models." + model_args.model).Model
    args_ckpt = Path(ckpt_path)
    assert args_ckpt.exists(), "Selected checkpoint does not exist!"
    net_model = net_module.load_from_checkpoint(args_ckpt, hparams=model_args, strict=strict)
    return net_model.eval()


def get_parser():
    parser = exp.ArgumentParserX(base_config_path='configs/fvdb_example_data/train_diffusion_16x16x16_dense.yaml', parents=[get_default_parser()])
    parser.add_argument('--coarse_or_fine', type=str, default='coarse', choices=['coarse', 'fine'], help='Coarse or fine VAE.')

    # parser.add_argument('--nosync', action='store_true', help='Do not synchronize nas even if forced.')
    parser.add_argument('--output_root', type=str, default="./outputs/diffusion_fvdb_example_data/", help='Output directory.')
    parser.add_argument('--suffix', type=str, default="", help='Suffix for output directory.')
    parser.add_argument('--hyper', type=str, default=None, help='Hyper parameters.')

    return parser

@torch.inference_mode()
def diffuse_vae_and_visualize(net_model, dataloader, saving_dir, known_args):
    ps.init()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = batch2device(batch, net_model.device)

        # 1. extract the latent code, as well as the ground truth hash tree
        clean_latents = net_model.extract_latent(batch)
        gt_tree = net_model.get_hash_tree(batch)
        # 2. sample the latent code
        sample_latents = net_model.random_sample_latents(clean_latents.grid, use_ddim=net_model.hparams.use_ddim, ddim_step=100, cond_dict={})['latents'] 
        decoded_res, _ = net_model.vae_decode(sample_latents)
        # 
        output_dict = {
            "gt_tree": gt_tree,
            "tree": decoded_res.structure_grid,
            "normal_features": decoded_res.normal_features,
        }

        vis_path = os.path.join(saving_dir, f"data_{batch_idx:03d}")

        # 1. compare the gt_tree and out_tree with gt_normal and out_normal
        visualize_structure_recon(output_dict, tmp_dir=vis_path, saving_dir=vis_path, save=True, fname_prefix="diffusion")
        visualize_normal_recon(output_dict, batch, tmp_dir=vis_path, saving_dir=vis_path, save=True, fname_prefix="diffusion")
    pass

def main():
    known_args = get_parser().parse_known_args()[0]
    if known_args.suffix != "":
        known_args.suffix = "_" + known_args.suffix
    saving_dir = Path(os.path.join(known_args.output_root, known_args.coarse_or_fine))
    print(f"Saving to {saving_dir}")
    saving_dir.mkdir(parents=True, exist_ok=True)

    # setup model config and path
    if known_args.coarse_or_fine == 'coarse':
        config_path = "configs/fvdb_example_data/train_diffusion_16x16x16_dense.yaml"
        ckpt_path = "/root/SCube/checkpoints/fvdb_example_data/coarse_diffusion/last.ckpt"

    elif known_args.coarse_or_fine == 'fine':
        config_path = "configs/fvdb_example_data/train_diffusion_16x16x16_dense.yaml"
        ckpt_path = "/root/SCube/checkpoints/fvdb_example_data/diffusion_dense/last.ckpt"
        raise NotImplementedError("Fine diffusion is not implemented yet. Especially on the cond_dict part")


    net_model = create_model_from_args(config_path, ckpt_path).cuda()

    dataset_kwargs = net_model.hparams.train_kwargs
    dataset_kwargs['batch_size'] = 1
    dataloader = net_model.train_dataloader() 

    diffuse_vae_and_visualize(net_model, dataloader, saving_dir, known_args)


if __name__ == "__main__":
    main()