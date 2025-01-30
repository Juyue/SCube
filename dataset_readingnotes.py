

import pytorch_lightning as pl
import torch
import wandb
import yaml
from omegaconf import OmegaConf
from packaging import version
from pycg import exp
from scube.data.base import DatasetSpec as DS

import scube.data as dataset

def get_dataset_spec(model_name, hparams):
    if model_name == 'autoencoder':
        all_specs = [DS.SHAPE_NAME, DS.INPUT_PC,
                     DS.GT_DENSE_PC, DS.GT_GEOMETRY]
        if hparams.use_input_normal:
            all_specs.append(DS.TARGET_NORMAL)
            all_specs.append(DS.GT_DENSE_NORMAL)
        if hparams.use_input_semantic or hparams.with_semantic_branch:
            all_specs.append(DS.GT_SEMANTIC)
        if hparams.use_input_intensity:
            all_specs.append(DS.INPUT_INTENSITY)

    elif model_name == 'diffusion':
        all_specs = get_dataset_spec('autoencoder', hparams)
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
        
        if hparams.use_sup_depth and hparams.sup_depth_type == 'rectified_metric3d_depth':
            all_specs.append(DS.IMAGES_DEPTH_MONO_EST_RECTIFIED)
        if hparams.use_sup_depth and hparams.sup_depth_type == 'lidar_depth':
            all_specs.append(DS.IMAGES_DEPTH_LIDAR_PROJECT)
        if hparams.use_sup_depth and hparams.sup_depth_type == 'depth_anything_v2_depth_inv':
            all_specs.append(DS.IMAGES_DEPTH_ANYTHING_V2_DEPTH_INV)
        if hparams.use_sup_depth and hparams.sup_depth_type == 'voxel_depth':
            pass # voxel depth is generated on the fly

    return all_specs

    

        # Note:
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
        hparams.val_dataset, get_dataset_spec('autoencoder', hparams), hparams, hparams.val_kwargs
    )
    x = next(iter(val_set))
    for key_name in x.keys():
        print(f"key_name: {key_name}, type: {type(x[key_name])}")
    import pdb; pdb.set_trace()

    # train_set = dataset.build_dataset(
    #     hparams.train_dataset, get_dataset_spec('autoencoder', hparams), hparams, hparams.train_kwargs
    # )
    # x = next(iter(train_set))
    # import pdb; pdb.set_trace()
    # print(x)