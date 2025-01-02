# path management 
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import 
import point_cloud_utils as pcu
import numpy as np
from tqdm import tqdm
import fvdb
import torch
import argparse

args = argparse.ArgumentParser()
args.add_argument('--data_root', type=str, default='/root/datasets/fvdb_example_data/meshes')
args.add_argument('--target_root', type=str, default='/root/datasets/fvdb_example_data/xcube')
args.add_argument('--num_vox', type=int, default=512) # 4096=16x16x16, 512=8x8x8, 64=4x4x4, 8=2x2x2, 1=1x1x1
args = args.parse_args()

data_root = args.data_root
target_root = args.target_root
num_vox = args.num_vox
target_dir = os.path.join(target_root, "%s" % str(num_vox))
os.makedirs(target_dir, exist_ok=True)

_mesh_names = [f.split(".")[0] for f in os.listdir(data_root) if os.path.isfile(os.path.join(data_root, f)) and f.endswith('.ply')]

if num_vox > 512:
    max_num_vox = num_vox
    sample_pcs_num = 5_000_000
else:
    max_num_vox = 512
    sample_pcs_num = 1_000_000
vox_size = 1.0 / max_num_vox


for mesh_name in _mesh_names:
    if mesh_name == "bunny" or mesh_name == "dragon":
        # dragon -> ooo
        # bunny -> no faceid info 
        continue
    mesh_path = os.path.join(data_root, f"{mesh_name}.ply")
    target_path = os.path.join(target_dir, f"{mesh_name}.pkl")
    print(f"Processing {mesh_name}")

    v, f = pcu.load_mesh_vf(mesh_path)
    fid, bc = pcu.sample_mesh_random(v, f, sample_pcs_num) # (N, 1), (N, 3)
    ref_xyz = pcu.interpolate_barycentric_coords(f, fid, bc, v) # (N, 3)

    n = pcu.estimate_mesh_face_normals(v, f) # (F, 3)
    ref_normal = n[fid] # (N, 3)

    # The function returns ijk, which contains the integer coordinates of the occupied voxels in the grid. Each row in ijk represents the (i,j,k) indices of an occupied voxel.
    ijk = pcu.voxelize_triangle_mesh(v, f.astype(np.int32), vox_size, np.zeros(3))
    grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor([torch.from_numpy(ijk).cuda()]), voxel_sizes=vox_size, origins=[vox_size / 2.] * 3)

    ref_xyz = torch.from_numpy(ref_xyz).float().cuda()
    ref_normal = torch.from_numpy(ref_normal).float().cuda()

    
    input_normal = grid.splat_trilinear(fvdb.JaggedTensor(ref_xyz), fvdb.JaggedTensor(ref_normal))
    input_normal.jdata /= (input_normal.jdata.norm(dim=1, keepdim=True) + 1e-6) # avoid nan

    
    ## normalize xyz to conv-onet scale; TODO: check out what is conv-onet scale
    xyz = grid.grid_to_world(grid.ijk.float()).jdata
    xyz_norm = xyz * 128 / 100
    ref_xyz = ref_xyz * 128 / 100
        
    # convert to fvdb_grid format
    if num_vox == 512:
        # not splatting
        target_voxel_size = 0.0025
        target_grid = fvdb.gridbatch_from_points(
                fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    elif num_vox == 16:
        # splatting
        target_voxel_size = 0.08
        target_grid = fvdb.gridbatch_from_nearest_voxels_to_points(
                    fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    elif num_vox == 128:
        # splatting
        target_voxel_size = 0.01
        target_grid = fvdb.gridbatch_from_nearest_voxels_to_points(
                    fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    elif num_vox == 256:
        target_voxel_size = 0.005
        target_grid = fvdb.gridbatch_from_nearest_voxels_to_points(
                    fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    elif num_vox == 1024:
        target_voxel_size = 0.00125
        target_grid = fvdb.gridbatch_from_points(
                    fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    else:
        raise NotImplementedError
        
    # get target normal
    target_normal = target_grid.splat_trilinear(fvdb.JaggedTensor(ref_xyz), fvdb.JaggedTensor(ref_normal))
    target_normal.jdata /= (target_normal.jdata.norm(dim=1, keepdim=True) + 1e-6)
    
    save_dict = {
        "points": target_grid.to("cpu"),
        "normals": target_normal.cpu(),
        "ref_xyz": ref_xyz.cpu(),
        "ref_normal": ref_normal.cpu(),
    }
        
    torch.save(save_dict, target_path)
    print(f"Saved to {target_path}")