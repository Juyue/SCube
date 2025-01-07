# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import polyscope as ps
import argparse
import torch
import trimesh
import numpy as np
import point_cloud_utils as pcu

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--path", type=str, required=True)
parser.add_argument('-i', "--id", type=int, default=0)
parser.add_argument('-v', "--vis_dir", type=str, default="outputs/visualize_object")
args = parser.parse_args()

# load result
result_dict_path = os.path.join(args.path, f"result_dict_{args.id}.pkl")
result_dict = torch.load(result_dict_path)

# coarse stage
coarse_xyz = result_dict["coarse_xyz"]
coarse_normal = result_dict["coarse_normal"]
coarse_normal_color = coarse_normal * 0.5 + 0.5

# fine stage
fine_xyz = result_dict["fine_xyz"]
fine_normal = result_dict["fine_normal"]
fine_normal = fine_normal / (np.linalg.norm(fine_normal, axis=1, keepdims=True) + 1e-6)

import pdb; pdb.set_trace()
vis_dir = os.path.join(args.vis_dir, f"result_dict_{args.id}")
os.makedirs(vis_dir, exist_ok=True)
ps.init()
ps.set_ground_plane_mode("none")

pc = ps.register_point_cloud(f"Coarse Point", coarse_xyz)
pc.add_color_quantity("normal", coarse_normal_color, enabled=True)
ps.screenshot(os.path.join(vis_dir, f"coarse_pointcloud.png"))

# ps.look_at((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
# ps.screenshot(os.path.join(vis_dir, f"coarse_pointcloud_zp1.png"))

# ps.look_at((0.0, 0.0, -1.0), (0.0, 0.0, 0.0))
# ps.screenshot(os.path.join(vis_dir, f"coarse_pointcloud_zn1.png"))

# ps.look_at((0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
# ps.screenshot(os.path.join(vis_dir, f"coarse_pointcloud_yp1.png"))

# ps.look_at((0.0, -1.0, 0.0), (0.0, 0.0, 0.0))
# ps.screenshot(os.path.join(vis_dir, f"coarse_pointcloud_yn1.png"))

# ps.look_at((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
# ps.screenshot(os.path.join(vis_dir, f"coarse_pointcloud_xp1.png"))

# ps.look_at((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
# ps.screenshot(os.path.join(vis_dir, f"coarse_pointcloud_xn1.png"))
ps.remove_all_structures()
pc = ps.register_point_cloud(f"Fine Point", fine_xyz)
fine_normal_color = fine_normal * 0.5 + 0.5
pc.add_color_quantity("normal", fine_normal_color, enabled=True)
ps.screenshot(os.path.join(vis_dir, f"fine_pointcloud.png"))

# # Mesh
# mesh = trimesh.load(os.path.join(args.path, f"mesh/mesh_{args.id}.obj"))
# mesh_n = pcu.estimate_mesh_vertex_normals(mesh.vertices, mesh.faces)
# mesh_c = (mesh_n + 1) / 2

# ps.register_surface_mesh(f"NKSR", mesh.vertices, mesh.faces).add_color_quantity("normal", mesh_c, enabled=True) 
# # ps.show()
# ps.screenshot(f"outputs/visualize_object/mesh_{args.id}.png")
