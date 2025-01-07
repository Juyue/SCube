import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from pycg import color
import os 
import polyscope as ps
import imageio.v3 as imageio
import torchvision
import cv2
import point_cloud_utils as pcu
import fvdb
from scube.data.base import DatasetSpec as DS

def vis_pcs(pcl_lst, S=3, vis_order=[2,0,1], bound=1):
    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    num_col = len(pcl_lst)
    for idx, pts in enumerate(pcl_lst):
        ax1 = fig.add_subplot(1, num_col, 1 + idx, projection='3d')
        rgb = None
        psize = S 
        
        # normalize the points
        if pts.size > 0:
            if np.abs(pts).max() > bound:
                pts = pts / np.abs(pts).max()
        
        ax1.scatter(pts[:, vis_order[0]], -pts[:, vis_order[1]], pts[:, vis_order[2]], s=psize, c=rgb)
        ax1.set_xlim(-bound, bound)
        ax1.set_ylim(-bound, bound)
        ax1.set_zlim(-bound, bound)
        ax1.grid(False)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig) # close the figure to avoid memory leak
    return image_from_plot


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


WAYMO_CATEGORY_NAMES = [
    "UNDEFINED", "CAR", "TRUCK", "BUS", "OTHER_VEHICLE", "MOTORCYCLIST", "BICYCLIST", "PEDESTRIAN",
    "SIGN", "TRAFFIC_LIGHT", "POLE", "CONSTRUCTION_CONE", "BICYCLE", "MOTORCYCLE", "BUILDING",
    "VEGETATION", "TREE_TRUNK", "CURB", "ROAD", "LANE_MARKER", "OTHER_GROUND", "WALKABLE", "SIDEWALK"
]

WAYMO_MAPPED = {
    0: ["SIGN", "TRAFFIC_LIGHT", "POLE", "CONSTRUCTION_CONE", "UNDEFINED"],
    1: ["MOTORCYCLIST", "BICYCLIST", "PEDESTRIAN", "BICYCLE", "MOTORCYCLE"],
    3: ["CAR", "TRUCK", "BUS", "OTHER_VEHICLE"],
    5: ["CURB", "LANE_MARKER"],
    4: ["VEGETATION", "TREE_TRUNK"],
    2: ["WALKABLE", "SIDEWALK"],
    6: ["BUILDING"],
    7: ["ROAD", "OTHER_GROUND"],
}


def get_waymo_palette():
    # Mapping from 23 Waymo categories to the 8 semantic visualization types
    # waymo_mapping key is waymo's semantic category. (23 categories)
    # waymo_mapping value is visualization category. (8 categories)
    waymo_mapping = np.zeros(23, dtype=np.int32)
    for visualization_type, waymo_semantic_categories in WAYMO_MAPPED.items():
        for waymo_semantic_name in waymo_semantic_categories:
            waymo_mapping[WAYMO_CATEGORY_NAMES.index(waymo_semantic_name)] = visualization_type

    waymo_palette = color.get_cmap_array('Set2')
    # Change the purple and green color
    waymo_palette[3] = color.get_cmap_array('Set3')[9]
    waymo_palette[4] = color.get_cmap_array('Set1')[2]

    return waymo_palette, waymo_mapping

waymo_palette, waymo_mapping = get_waymo_palette()

#################################################
# Visualization during training
#################################################

def _visualize_grid_and_feature(grid, feature=None, grid_idx=0,  vis_path="../tree_vis.png", save=True):

    if feature is None:
        feature = grid.grid_to_world(grid.ijk.float())

    ps.remove_all_structures()
    if grid.num_voxels[grid_idx] > 0:
        feature.jdata = (feature.jdata - feature.jdata.min(dim=0).values) / (
            feature.jdata.max(dim=0).values - feature.jdata.min(dim=0).values
        )
        grid_color = feature[grid_idx].jdata[:, :3].cpu().numpy().repeat(8, axis=0).reshape(-1, 3)

        grid_mesh = pcu.voxel_grid_geometry(grid.ijk[grid_idx].jdata.cpu().numpy(), 0.1, grid.voxel_sizes[grid_idx].cpu().numpy())

        ps.register_surface_mesh("grid", grid_mesh[0], grid_mesh[1], enabled=True).add_color_quantity(
            "color", grid_color, enabled=True
        )
    ps.screenshot(vis_path)

def visualize_structure_recon(output_dict, tmp_dir = None, saving_dir=None, save=True, fname_prefix=None):
    if tmp_dir is None:
        tmp_dir = os.path.join("./outputs/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    ps.init()

    # 0. compare the gt_tree and out_tree?
    n_depth = len(output_dict['tree'].keys())
    n_grid = output_dict['tree'][0].grid_count

    grid_imgs = []
    for grid_idx in range(n_grid):
        out_tree_files = {}
        gt_tree_files = {}

        for tree_depth in output_dict['tree'].keys():
            out_tree = output_dict['tree'][tree_depth]
            out_tree_file = os.path.join(tmp_dir, f"{fname_prefix}_grid_{grid_idx}_out_tree_depth_{tree_depth}.png")
            _visualize_grid_and_feature(out_tree, grid_idx=grid_idx, vis_path=out_tree_file)
            out_tree_files[tree_depth] = out_tree_file

            if "gt_tree" in output_dict:
                gt_tree = output_dict['gt_tree'][tree_depth]
                gt_tree_file = os.path.join(tmp_dir, f"{fname_prefix}_grid_{grid_idx}_gt_tree_depth_{tree_depth}.png")
                _visualize_grid_and_feature(gt_tree, grid_idx=grid_idx, vis_path=gt_tree_file)
                gt_tree_files[tree_depth] = gt_tree_file
        # tile them together
        out_image_list = []
        gt_image_list = []
        for tree_depth in output_dict['tree'].keys():

            out_image = imageio.imread(out_tree_files[tree_depth])
            out_image_list.append(out_image)

            if "gt_tree" in output_dict:
                gt_image = imageio.imread(gt_tree_files[tree_depth])
                gt_image_list.append(gt_image)

        if "gt_tree" in output_dict:
            image_list = [torch.tensor(img) for img in out_image_list +  gt_image_list]
        else:
            image_list = [torch.tensor(img) for img in out_image_list]

        image_list = torch.stack(image_list)
        grid_img = torchvision.utils.make_grid(image_list.permute(0, 3, 1, 2), nrow=n_depth, padding=2)

        if save:
            os.makedirs(saving_dir, exist_ok=True)
            imageio.imwrite(os.path.join(saving_dir, f"{fname_prefix}_grid_{grid_idx}_structure_tree.png"), grid_img.permute(1, 2, 0).numpy())

        # 4. remove the out_tree and gt_tree_depth
        for file in out_tree_files.values():
            os.remove(file)
        for file in gt_tree_files.values():
            os.remove(file)
        
        grid_imgs.append(grid_img)
    return grid_imgs

def visualize_normal_recon(output_dict, batch, tmp_dir=None, saving_dir=None, save=True, fname_prefix=None):
    if tmp_dir is None:
        tmp_dir = os.path.join("./outputs/tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    n_grid = output_dict['tree'][0].grid_count
    normal_features = output_dict["normal_features"]
    if len(normal_features) == 0:
        # TODO: figure out why normal_features is empty?
        return []

    normal_images = []
    for grid_idx in range(n_grid):
        if len(normal_features) > 0:
            pred_grid = normal_features[-1].grid
            pred_normal = normal_features[-1].data
            pred_normal_file = os.path.join(tmp_dir, f"{fname_prefix}_grid_{grid_idx}_pred_normal.png")
            _visualize_grid_and_feature(pred_grid, pred_normal, grid_idx=grid_idx, vis_path=pred_normal_file, save=save)
        else:
            ps.remove_all_structures()
            ps.screenshot(pred_normal_file)

        if "gt_tree" in output_dict:
            gt_grid = batch[DS.INPUT_PC] # GridBatch
            gt_normal = batch[DS.TARGET_NORMAL] # list of tensors
            gt_normal = fvdb.JaggedTensor(gt_normal)
            gt_normal_file = os.path.join(tmp_dir, f"{fname_prefix}_grid_{grid_idx}_gt_normal.png")
            _visualize_grid_and_feature(gt_grid, gt_normal, grid_idx=grid_idx, vis_path=gt_normal_file, save=save)

        # Write text on the left corner of the gt_normal_image
        pred_normal_image = imageio.imread(pred_normal_file)
        if "gt_tree" in output_dict:
            gt_normal_image = imageio.imread(gt_normal_file)
        else:
            gt_normal_image = np.zeros_like(pred_normal_image)

        tmp = [torch.tensor(pred_normal_image), torch.tensor(gt_normal_image)]
        tmp = torch.stack(tmp)
        grid_img = torchvision.utils.make_grid(tmp.permute(0, 3, 1, 2), nrow=1, padding=2)

        if save:
            os.makedirs(saving_dir, exist_ok=True)
            imageio.imwrite(os.path.join(saving_dir, f"{fname_prefix}_grid_{grid_idx}_normal.png"), grid_img.permute(1, 2, 0).numpy())

        normal_images.append(grid_img)

        os.remove(pred_normal_file)
        if "gt_tree" in output_dict:
            os.remove(gt_normal_file)
    
    return normal_images
    


    


