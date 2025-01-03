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
        self.files = [os.path.join(onet_base_path, f) for f in files]

    def __len__(self):
        return len(self.files)
            
    def _get_item(self, data_id, rng):
        data_id = data_id % len(self.files)

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

        return data
