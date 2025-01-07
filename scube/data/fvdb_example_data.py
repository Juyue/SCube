# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
from pathlib import Path

import fvdb
import torch
from loguru import logger

from scube.data.base import DatasetSpec as DS
from scube.data.base import RandomSafeDataset

import polyscope as ps

class FVDBExampleDataDataset(RandomSafeDataset):
    def __init__(self, onet_base_path, spec, resolution, random_seed=0, skip_on_error=False, **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)

        self.resolution = resolution
        onet_base_path = os.path.join(onet_base_path, str(resolution))
        self.spec = spec

        # build index to file mapping.
        files = os.listdir(onet_base_path)
        self.files = [os.path.join(onet_base_path, f) for f in files if "car" in f]

    def __len__(self):
        return len(self.files)
            
    def _get_item(self, data_id, rng):
        data_id = data_id % len(self.files)
        shape_name = self.files[data_id].split("/")[-1].split(".")[0]

        data = {}
        input_data = torch.load(self.files[data_id])

        input_points = input_data['points']
        input_normals = input_data['normals'].jdata
    
        if DS.TARGET_NORMAL in self.spec:
            data[DS.TARGET_NORMAL] = input_normals
    
        if DS.INPUT_PC in self.spec:
            data[DS.INPUT_PC] = input_points
                
        if DS.GT_DENSE_PC in self.spec:
            data[DS.GT_DENSE_PC] = input_points

        if DS.GT_DENSE_NORMAL in self.spec:
            data[DS.GT_DENSE_NORMAL] = input_normals
        
        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = shape_name
        

        return data

def _visualize_pc(data, vis_dir="outputs/fvdb_example_data/visualize", name="pc"):

    grid = data[DS.INPUT_PC]
    ijk = grid.ijk.jdata
    xyz = grid.grid_to_world(ijk.float().view(-1, 3)).jdata

    normal = data[DS.TARGET_NORMAL]
    normal_color = normal * 0.5 + 0.5
    # import pdb; pdb.set_trace()

    print("="*100)
    print(data[DS.SHAPE_NAME])
    print(xyz.shape)


    ps.remove_all_structures()
    pc = ps.register_point_cloud(name, xyz)
    pc.add_color_quantity("normal", normal_color, enabled=True)
    ps.screenshot(os.path.join(vis_dir, f"{name}.png"))

def _visualize_dataset(dataset, vis_dir="outputs/fvdb_example_data/visualize"):
    os.makedirs(vis_dir, exist_ok=True)
    ps.init()
    for i in range(min(5, len(dataset))):
        data = dataset[i]
        _visualize_pc(data, vis_dir, name=f"{data[DS.SHAPE_NAME]}_{dataset.resolution}")

if __name__ == "__main__":
    # load data and visualize them 
    root_dir = "/root/datasets/fvdb_example_data/xcube"
    spec = [DS.INPUT_PC, DS.TARGET_NORMAL, DS.GT_DENSE_PC, DS.GT_DENSE_NORMAL, DS.SHAPE_NAME]

    dataset_128 = FVDBExampleDataDataset(
        onet_base_path=root_dir,
        spec=spec,
        resolution=128,
    )
    _visualize_dataset(dataset_128)

    dataset_512 = FVDBExampleDataDataset(
        onet_base_path=root_dir,
        spec=spec,
        resolution=512,
    )
    _visualize_dataset(dataset_512)