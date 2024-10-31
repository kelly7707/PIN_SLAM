import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.neural_points import NeuralPoints
from model.attention import Attention
from utils.config import Config
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import torch
import open3d as o3d
import numpy as np
import wandb
from datetime import datetime
from torch import optim
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import yaml

from Dataset_Collection import PointCloudDataset, MultiBagBatchSampler
from dataset.slam_dataset import crop_frame, create_homogeneous_transform
from dataset.slam_dataset import SLAMDataset as OriginalSLAMDataset
from utils.mesher import Mesher
from utils.tools import voxel_down_sample_torch, setup_wandb, split_chunks, get_gradient, readVariable
from utils.mapper import Mapper
from utils.visualizer import MapVisualizer
from utils.loss import *


class SLAMDataset(Dataset):
    def __init__(self, config: Config) -> None:

        super().__init__()

        self.config = config
        self.silence = config.silence
        self.dtype = config.dtype
        self.device = config.device

        self.stop_status = False
        self.gt_pose_provided = True
        self.processed_frame: int = 0
        self.travel_dist = []
        self.frame_normal_torch = None

        self.gt_poses = []
        self.T_Wl_Llast = np.eye(4)
        self.last_odom_tran = np.eye(4)
        self.T_Wl_Lcur = np.eye(4)

        # # visual
        # # current frame point cloud (for visualization)
        # self.cur_frame_o3d = o3d.geometry.PointCloud()
        # # current frame bounding box in the world coordinate system
        # self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()
        # # merged downsampled point cloud (for visualization)
        # self.map_down_o3d = o3d.geometry.PointCloud()
        # # map bounding box in the world coordinate system
        # self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()
        # self.static_mask = None

    def clear(self):
        self.stop_status = False
        self.gt_pose_provided = True
        self.processed_frame: int = 0
        self.travel_dist = []
        self.frame_normal_torch = None

        self.gt_poses = []
        self.T_Wl_Llast = np.eye(4)
        self.last_odom_tran = np.eye(4)
        self.T_Wl_Lcur = np.eye(4)

        # visual
        # current frame point cloud (for visualization)
        self.cur_frame_o3d = o3d.geometry.PointCloud()
        # current frame bounding box in the world coordinate system
        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()
        # merged downsampled point cloud (for visualization)
        self.map_down_o3d = o3d.geometry.PointCloud()
        # map bounding box in the world coordinate system
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()
        self.static_mask = None
    
    def update_o3d_map(self):
        # Use the method from the original class
        return OriginalSLAMDataset.update_o3d_map(self)
    
    def dataset_process(self, point_cloud, cur_pose_torch):
        # -- poses
        self.T_Wl_Lcur = cur_pose_torch.detach().cpu().numpy()
        self.last_tran = np.linalg.inv(self.T_Wl_Llast) @ self.T_Wl_Lcur
        if self.processed_frame == 0:
            self.travel_dist.append(0.)
        else:
            cur_frame_travel_dist = np.linalg.norm(self.last_tran[:3,3])
            if cur_frame_travel_dist > self.config.surface_sample_range_m * 40.0: # too large translation in one frame --> lose track
                self.lose_track = True 
                # sys.exit("Too large translation in one frame, system failed") # FIXME
               
            accu_travel_dist = self.travel_dist[-1] + cur_frame_travel_dist
            self.travel_dist.append(accu_travel_dist)
        self.T_Wl_Llast = self.T_Wl_Lcur
        
        # -- point cloud
        # SLAMDataset.read_frame_ros # TODO: point_cloud is already a tensor? dtype consistent?
        self.cur_point_cloud_torch = point_cloud.to(device=self.device, dtype=self.dtype)
        # self.cur_point_cloud_torch = torch.tensor(point_cloud, device=self.device, dtype= self.dtype)
        # SLAMDataset.preprocess_frame
        if self.config.adaptive_range_on:
            pc_max_bound, _ = torch.max(self.cur_point_cloud_torch[:, :3], dim=0)
            pc_min_bound, _ = torch.min(self.cur_point_cloud_torch[:, :3], dim=0)

            min_x_range = min(torch.abs(pc_max_bound[0]),  torch.abs(pc_min_bound[0]))
            min_y_range = min(torch.abs(pc_max_bound[1]),  torch.abs(pc_min_bound[1]))
            max_x_y_min_range = max(min_x_range, min_y_range)

            crop_max_range = min(self.config.max_range, 2.*max_x_y_min_range)
        else:
            crop_max_range = self.config.max_range

        train_voxel_m = (crop_max_range/self.config.max_range) * self.config.vox_down_m #  input downsample 
        idx = voxel_down_sample_torch(self.cur_point_cloud_torch[:,:3], train_voxel_m)
        self.cur_point_cloud_torch = self.cur_point_cloud_torch[idx]

        cur_point_ts_torch = None
        self.cur_point_cloud_torch, _ = crop_frame(self.cur_point_cloud_torch, cur_point_ts_torch, 
                                                self.config.min_z, self.config.max_z, 
                                                self.config.min_range, crop_max_range)
        
        # --- estimate_normals direction for sdf 
        if self.config.estimate_normal:
            cur_point_cloud_np = self.cur_point_cloud_torch.detach().cpu().numpy()

            o3d_device = o3d.core.Device("CPU:0") # cuda:0
            o3d_dtype = o3d.core.float32
            cur_point_cloud_o3d = o3d.t.geometry.PointCloud(o3d_device)
            cur_point_cloud_o3d.point["positions"] = o3d.core.Tensor(
                cur_point_cloud_np, o3d_dtype, o3d_device
            )
            print("Estimate normal")
            cur_point_cloud_o3d.estimate_normals(max_nn=20)
            cur_point_cloud_o3d.orient_normals_towards_camera_location() # orient normals towards the default origin(0,0,0).
            self.frame_normal_torch = torch.tensor(cur_point_cloud_o3d.point.normals.numpy(), dtype=self.dtype, device=self.device)



def pin_slam_visual(config, geo_mlp, o3d_vis, slamdataset, neural_points, mapper, mesher):
     # --- Mesh reconstruction and visualization

    geo_mlp.eval()
    o3d_vis.cur_frame_id = slamdataset.processed_frame # frame id in the data folder

    slamdataset.static_mask = mapper.static_mask
    slamdataset.cur_sem_labels_torch = None
    slamdataset.update_o3d_map()
            # if config.track_on and slamdataset.processed_frame > 0  and (weight_pc_o3d is not None): 
            #     slamdataset.cur_frame_o3d = weight_pc_o3d

    neural_pcd = None
    if o3d_vis.render_neural_points: # or (frame_id == last_frame): # last frame also vis
        neural_pcd = neural_points.get_neural_points_o3d(query_global=o3d_vis.vis_global, color_mode=o3d_vis.neural_points_vis_mode, random_down_ratio=1) # select from geo_feature, ts and certainty

    # reconstruction by marching cubes
    mesher.ts = slamdataset.processed_frame # deprecated
    cur_mesh = None
    # if config.mesh_freq_frame > 0:
    #     if o3d_vis.render_mesh and (slamdataset.processed_frame == 0 or (slamdataset.processed_frame+1) % config.mesh_freq_frame == 0):     #  or frame_id == last_frame         
            
    #         # update map bbx
    #         global_neural_pcd_down = neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=23) # prime number
    #         slamdataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()
            
    #         mesh_path = None # no need to save the mesh

    #         # figure out how to do it efficiently
    #         if config.mc_local or (not o3d_vis.vis_global): # only build the local mesh
    #             # cur_mesh = mesher.recon_aabb_mesh(slamdataset.cur_bbx, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)
    #             chunks_aabb = split_chunks(global_neural_pcd_down, slamdataset.cur_bbx, o3d_vis.mc_res_m*100) # reconstruct in chunks
    #             cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
    #         else:
    #             aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
    #             chunks_aabb = split_chunks(global_neural_pcd_down, aabb, o3d_vis.mc_res_m*300) # reconstruct in chunks
    #             cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
    cur_sdf_slice = None
    if config.sdfslice_freq_frame > 0:
        if o3d_vis.render_sdf and (slamdataset.processed_frame == 0 or (slamdataset.processed_frame+1) % config.sdfslice_freq_frame == 0):
            slice_res_m = config.voxel_size_m * 0.2
            sdf_bound = config.surface_sample_range_m * 4.
            query_sdf_locally = True
            if o3d_vis.vis_global:
                cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(slamdataset.map_bbx, slamdataset.T_Wl_Lcur[2,3]+o3d_vis.sdf_slice_height, slice_res_m, False, -sdf_bound, sdf_bound) # horizontal slice
            else:
                cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(slamdataset.cur_bbx, slamdataset.T_Wl_Lcur[2,3]+o3d_vis.sdf_slice_height, slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # horizontal slice (local)
            if config.vis_sdf_slice_v:
                cur_sdf_slice_v = mesher.generate_bbx_sdf_ver_slice(slamdataset.cur_bbx, slamdataset.T_Wl_Lcur[0,3], slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # vertical slice (local)
                cur_sdf_slice = cur_sdf_slice_h + cur_sdf_slice_v
            else:
                cur_sdf_slice = cur_sdf_slice_h
                        
    pool_pcd = mapper.get_data_pool_o3d(down_rate=17, only_cur_data=o3d_vis.vis_only_cur_samples) if o3d_vis.render_data_pool else None # down rate should be a prime number
    loop_edges = None
    o3d_vis.update_traj(slamdataset.T_Wl_Lcur, None, slamdataset.gt_poses, None, loop_edges)
    o3d_vis.update(slamdataset.cur_frame_o3d, slamdataset.T_Wl_Lcur, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)
            

def print_mlp_weights(print_content,mlp_decoder):
    print(f'{print_content}')
    weights = {}
    for name, param in mlp_decoder.named_parameters():
        # print(f"{name}: {param.data}")
        weights[name] = param.data
    print(weights[name])
    # return weights


def setup_optimizer_pretrain(config: Config, neural_point_feat = None, mlp_geo_param = None, 
                    poses = None, lr_ratio = 1.0) -> Optimizer:
    lr_cur = config.lr * lr_ratio
    lr_pose = config.lr_pose
    weight_decay = config.weight_decay
    opt_setting = []
    # weight_decay is for L2 regularization, only applied to MLP
    if mlp_geo_param is not None: 
        mlp_geo_param_opt_dict = {'params': mlp_geo_param, 'lr': lr_cur, 'weight_decay': weight_decay} 
        opt_setting.append(mlp_geo_param_opt_dict)

    if poses is not None:
        poses_opt_dict = {'params': poses, 'lr': lr_pose, 'weight_decay': weight_decay}
        opt_setting.append(poses_opt_dict)
    
    # feature octree
    if neural_point_feat is not None:
        feat_opt_dict = {'params': neural_point_feat, 'lr': lr_cur, 'weight_decay': weight_decay} 
        opt_setting.append(feat_opt_dict)

    if config.opt_adam:
        opt = optim.Adam(opt_setting, betas=(0.9,0.99), eps = config.adam_eps)  #, set weight_decay=1e-2 in config
        # opt = optim.AdamW(opt_setting, betas=(0.9, 0.99), eps=config.adam_eps)
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)
    
    return opt 


def train_multi_sequences(sequences_pc_curframe, sequences_gt_poses_curframe, labels, slamdataset_list, neural_points_list, mapper_list, mesher_list, config, geo_mlp, o3d_vis):

    for index in range(len(labels)):
        point_cloud = sequences_pc_curframe[index] # example: torch.Size([131072, 3])
        # filter nan (padding during points collection, due to the different number of points in each frame)    
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[~nan_mask]

        cur_pose_torch = sequences_gt_poses_curframe[index]

        slamdataset = slamdataset_list[labels[index]]
        neural_points = neural_points_list[labels[index]]
        mapper = mapper_list[labels[index]]
        mesher = mesher_list[labels[index]]

        # # point_cloud = point_cloud.to(device=device, dtype=dtype) 
        # # cur_pose_torch.to(device=device) # TODO, dtype=dtype ? 
        slamdataset.dataset_process(point_cloud, cur_pose_torch)
        neural_points.travel_dist = torch.tensor(np.array(slamdataset.travel_dist), device = config.device, dtype=config.dtype) 

        cur_sem_labels_torch = None
        mapper.process_frame(slamdataset.cur_point_cloud_torch, cur_sem_labels_torch,
                                cur_pose_torch.detach().to(device=config.device), slamdataset.processed_frame, use_travel_dist= True)
    
    
    # Initialize optimizers
    geo_mlp_optimizer = setup_optimizer_pretrain(config, mlp_geo_param=list(geo_mlp.parameters()))
    neural_point_optimizers = {key: setup_optimizer_pretrain(config, neural_point_feat = list(neural_points.parameters())) for key, neural_points in neural_points_list.items()}

    # # --- for the first frame, we need more iterations to do the initialization (warm-up)
    # # cur_iter_num = int(config.iters * config.init_iter_ratio) if first_train else config.iters # 15
    iter_count = config.iters # 15
    for iter in tqdm(range(iter_count), disable = False):
        geo_mlp.train()
        # geo_mlp_optimizer.zero_grad()
        # Initialize the accumulated loss for geo_mlp update
        # accumulated_loss = 0.0

        for index in range(len(labels)):
            geo_mlp_optimizer.zero_grad()
            
            # mapper_list[labels[index]].pretrain_decoder(iter, neural_point_optimizers[index])
            # -- mapping
            cur_mapper = mapper_list[labels[index]]

            # we do not use the ray rendering loss here for the incremental mapping
            coord, sdf_label, ts, _, sem_label, color_label, weight = cur_mapper.get_batch(global_coord=not cur_mapper.ba_done_flag) # coord here is in global frame if no ba pose update

            poses = cur_mapper.used_poses[ts]
            origins = poses[:,:3,3]

            if cur_mapper.require_gradient: #default false
                coord.requires_grad_(True)

            geo_feature, color_feature, weight_knn, _, certainty = cur_mapper.neural_points.query_feature(coord, ts, query_color_feature=config.color_on) # query feature of neighbors

            sdf_pred = geo_mlp.sdf(geo_feature, coord) # predict the scaled sdf with the feature # [N, K, 1]
            sdf_pred = sdf_pred.squeeze(1)

            surface_mask = (torch.abs(sdf_label) < config.surface_sample_range_m)  # weight > 0

            if cur_mapper.require_gradient:
                g = get_gradient(coord, sdf_pred) # to unit m  
            elif config.numerical_grad: # use for mapping # do not use this for the tracking, still analytical grad for tracking   
                g = cur_mapper.get_numerical_gradient(coord[::config.gradient_decimation], 
                                                sdf_pred[::config.gradient_decimation],
                                                config.voxel_size_m*config.num_grad_step_ratio) #  


            # calculate the loss   
            sdf_individual_losses = torch.zeros_like(sdf_label)      
            eikonal_individual_losses = torch.zeros_like(sdf_label)

            cur_loss = 0.0
            weight = torch.abs(weight).detach() # weight's sign indicate the sample is around the surface or in the free space
            if config.main_loss_type == 'bce': # [used]
                sdf_loss, sdf_individual_losses = sdf_bce_loss(sdf_pred, sdf_label, cur_mapper.sdf_scale, weight, config.loss_weight_on) 
            else:
                sys.exit("Please choose a valid loss type")
            cur_loss += sdf_loss

            # # optional consistency regularization loss
            consistency_loss = 0.0

            # ekional loss
            eikonal_loss = 0.0
            if config.ekional_loss_on and config.weight_e > 0: # MSE with regards to 1  
                surface_mask_decimated = surface_mask[::config.gradient_decimation]
                if config.ekional_add_to == "freespace":
                    g_used = g[~surface_mask_decimated]
                elif config.ekional_add_to == "surface":
                    g_used = g[surface_mask_decimated]
                else: # "all"  # both the surface and the freespace, used here # [used]
                    g_used = g
                
                eikonal_individual_losses = (g_used.norm(2, dim=-1) - 1.0) ** 2
                
                eikonal_loss = eikonal_individual_losses.mean() # both the surface and the freespace
                cur_loss += config.weight_e * eikonal_loss

            # accumulated_loss += cur_loss.detach().item() 
            neural_point_optimizers[labels[index]].zero_grad(set_to_none=True)
            # cur_loss.backward(retain_graph=True)
            cur_loss.backward()

            # --- Monitoring gradients
            # Monitor gradients for geo_mlp
            count = 0
            geo_mlp_grads = {}
            for name, param in cur_mapper.geo_mlp.named_parameters():
                if param.grad is not None:
                    count += 1
                    grad_norm = param.grad.norm().item()
                    geo_mlp_grads[f'grad/geo_mlp/{name}'] = grad_norm
                    # assert grad_norm < 10 and grad_norm > 1e-19, f"Gradient norm for {name} is too large or too small"
                # assert count > 0, "No gradient computed"
            # Monitor gradients for neural_points
            neural_points_grads = {}
            count = 0
            for name, param in cur_mapper.neural_points.named_parameters():
                if param.grad is not None:
                    count += 1
                    grad_norm = param.grad.norm().item()
                    neural_points_grads[f'grad/neural_points/{name}'] = grad_norm
                    # assert grad_norm < 10 and grad_norm > 1e-9, f"Geo_feature Gradient norm for {name} is too large or too small"
                assert count > 0, "Geo_feature No gradient computed"
            
            neural_point_optimizers[labels[index]].step()

            cur_mapper.total_iter += 1

            if config.wandb_vis_on:
                wandb_log_content = {'iter': cur_mapper.total_iter, 'loss/total_loss': cur_loss, 'loss/sdf_loss': sdf_loss, \
                                        'loss/eikonal_loss': eikonal_loss, 'loss/consistency_loss': consistency_loss,
                                        } #, \
                                        #'loss/sem_loss': sem_loss, 'loss/color_loss': color_loss} 
                # Combine gradient logs into wandb log content
                if geo_mlp_grads:
                    wandb_log_content.update(geo_mlp_grads)
                if neural_points_grads:
                    wandb_log_content.update(neural_points_grads)
                wandb.log(wandb_log_content)

        # accumulated_loss_tensor = torch.tensor(accumulated_loss, requires_grad=True)
        # accumulated_loss_tensor.backward()  # Backpropagate accumulated loss for geo_mlp
            geo_mlp_optimizer.step() 


    # update the global map
    for index in range(len(labels)):
        neural_points_list[labels[index]].assign_local_to_global() 
        if config.wandb_vis_on:
            wandb_log_content = {'frame': slamdataset_list[labels[index]].processed_frame}
            wandb.log(wandb_log_content)

        slamdataset_list[labels[index]].processed_frame += 1

        if config.o3d_vis_on:
            # pin_slam_visual(config, geo_mlp, o3d_vis, slamdataset, neural_points, mapper, mesher)
            pin_slam_visual(config, geo_mlp, o3d_vis, slamdataset_list[labels[1]], neural_points_list[labels[1]], mapper_list[labels[1]], mesher_list[labels[1]])

    # # unique checking
    # print_mlp_weights('-------geo_mlp weight inside function',geo_mlp)
    # print_mlp_weights('-------geo_mlp weight inside class',cur_mapper.geo_mlp)


def main():
    # --------------------------------------------
    
    # -------------- Dataset Collection ----------
    # New college dataset
    NC_point_cloud_topic = "/os_cloud_node/points"
    nce_bag_path = 'data/Newer_College_Dataset/2021-07-01-10-37-38-quad-easy.bag'
    ncm_bag_path = 'data/Newer_College_Dataset/medium/2021-07-01-11-31-35_0-quad-medium.bag'
    nc_mine_bag_path = 'data/Newer_College_Dataset/mine_easy/2021-04-12-11-11-33-mine_medium.bag'
    nce_gt_pose_file = 'data/Newer_College_Dataset/gt-nc-quad-easy_TMU.csv'
    ncm_gt_pose_file = 'data/Newer_College_Dataset/medium/gt-nc-quad-medium.csv'
    nc_mine_gt_pose_file = 'data/Newer_College_Dataset/mine_easy/medium_gt_state_tum_corrected.csv'

    calib_file_path = 'data/Newer_College_Dataset/os_imu_lidar_transforms.yaml'
    with open(calib_file_path, 'r') as file:
        calibration_data = yaml.safe_load(file)
    os_sensor_to_base_data = calibration_data['os_sensor_to_base']
    translation_base = np.array(os_sensor_to_base_data['translation'])
    rotation_base = np.array(os_sensor_to_base_data['rotation'])

    T_GT_L_nc = create_homogeneous_transform(translation_base, rotation_base)
    T_L_GT_nc = np.linalg.inv(T_GT_L_nc)
    
    # ASL
    ASL_point_cloud_topic = "/ouster/points"
    field_bag_path = './data/ASL/field_s/2023-08-09-19-05-05-field_s.bag'
    katzensee_bag_path = 'data/ASL/katzensee/2023-08-21-10-20-22-katzensee_s.bag'
    katzensees_gt_pose_file = 'data/ASL/katzensee/gt-katzensee_s.csv'
    fields_gt_pose_file = 'data/ASL/field_s/gt-field_s.csv'

    # Kitti
    kitti360_point_cloud_topic = "/kitti360/cloud"
    kitti360_bag_path = 'data/kitti_360/2013_05_28_drive_0000.bag'
    kitti360_gt_pose_file = 'data/kitti_360/2013_05_28_drive_0000/data_poses/2013_05_28_drive_0000_sync/poses.txt'
    
        # calibration: https://www.cvlibs.net/datasets/kitti-360/documentation.php  & https://github.com/autonomousvision/kitti360Scripts/blob/d4d4885102f72ea8d96c9e72e2ff03a8834353a4/kitti360scripts/devkits/commons/loadCalibration.py#L92C1-L100C14 
    t_imu_cam_file = 'data/kitti_360/2013_05_28_drive_0000/calibration/calib_cam_to_pose.txt'
    t_velo_cam_file = 'data/kitti_360/2013_05_28_drive_0000/calibration/calib_cam_to_velo.txt'
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    with open(t_imu_cam_file, 'r') as fid:
        T_IMU_CAM =  np.concatenate((readVariable(fid, 'image_00', 3, 4), lastrow))# 
    T_Velo_CAM = np.concatenate((np.loadtxt(t_velo_cam_file).reshape(3,4), lastrow)) # image_00
    T_GT_L_kitti360 = T_IMU_CAM @ np.linalg.inv(T_Velo_CAM) # T_IMU_L 

    kitti_bag_path = 'data/kitti_example/sequences/00/bag00.bag'

    # list
    # ts_field_name = "t"
    sequence_paths = [f'{ncm_bag_path}', f'{kitti360_bag_path}', f'{nc_mine_bag_path}']
    point_cloud_topics = [NC_point_cloud_topic, kitti360_point_cloud_topic, NC_point_cloud_topic]
    gt_poses_files = [ncm_gt_pose_file, kitti360_gt_pose_file, nc_mine_gt_pose_file]
    gt_poses_trans = [T_GT_L_nc, T_GT_L_kitti360, T_GT_L_nc]

    # sequence_paths = [f'{nce_bag_path}', f'{ncm_bag_path}', f'{kitti360_bag_path}']
    # point_cloud_topics = [NC_point_cloud_topic, NC_point_cloud_topic, kitti360_point_cloud_topic]
    # gt_poses_files = [nce_gt_pose_file, ncm_gt_pose_file, kitti360_gt_pose_file]
    # gt_poses_trans = [T_GT_L_nc, T_GT_L_nc, T_GT_L_kitti360]

    # Create dataset with sequences from all bags
    dataset = PointCloudDataset(sequence_paths, gt_poses_files, point_cloud_topics, num_sequences_per_bag=4) # 8/10
    num_datasets = dataset.unique_bag_labels.shape[0]
    # Use custom batch sampler for one or two sequences from the same bag per batch
    batch_sampler = MultiBagBatchSampler(dataset, num_datasets=num_datasets, sequences_per_batch=1)
    # Dataloader with custom batch sampler
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    # --------------------------------------------
    config_path = "./config/lidar_slam/run_ros_general_pretrain.yaml"
    config = Config()
    config.load(config_path)

    if config.wandb_vis_on:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp
        run_name = config.name + "_" + ts  # modified to a name that is easier to index
        run_path = os.path.join(config.output_root, run_name)
        access = 0o755
        os.makedirs(run_path, access, exist_ok=True)
        # set up wandb
        setup_wandb()
        wandb.init(project="pin-slam-pretrain", config=vars(config), dir=run_path) # your own worksapce
        wandb_run_id = wandb.run.id
        wandb.run.name = "testing0 |" + run_name
        # Set a description for the run
        wandb.run.notes = "nce+ ncm - 2*5s sequences -- 2 sequences per batch"

    if config.o3d_vis_on:
        o3d_vis = MapVisualizer(config)
    else: 
        o3d_vis = None

    # set the random seed
    o3d.utility.random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed) 

    device = config.device
    dtype = config.dtype

    checkpoint_dir = 'pretrained_mlp_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # -- model    
    geo_mlp = Attention(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)

    # -- can also be moved inside train_single_sequence
    neural_points_list = {}
    slamdataset_list = {}
    mapper_list = {}
    mesher_list = {}
    for i in range(num_datasets):
        neural_points_list[dataset.unique_bag_labels[i]] = NeuralPoints(config)
        slamdataset_list[dataset.unique_bag_labels[i]] = SLAMDataset(config)
        mapper_list[dataset.unique_bag_labels[i]] = Mapper(config, slamdataset_list[dataset.unique_bag_labels[i]], neural_points_list[dataset.unique_bag_labels[i]], geo_mlp, None, None)
        if config.o3d_vis_on:
            mesher_list[dataset.unique_bag_labels[i]] = Mesher(config, neural_points_list[dataset.unique_bag_labels[i]], geo_mlp, None, None)
        else:
            mesher_list[dataset.unique_bag_labels[i]] = None


    # --------------------------------------------
    first_train = True
    idx_batch = 0
    for batch in dataloader:
        """Each batch contains sequences from all dataset (one sequence per dataset), train them in parallel.
        contains three components:
            1. Point cloud sequences (list of tensors) 
            2. Ground truth poses (list of tensors)
            3. Labels (list of bag file paths)"""

        # Unpacking the batch into sequences and labels
        sequences_pc, sequences_gt_poses, labels = batch
        print('-----------')
        print(sequences_pc.shape) # example: torch.Size([2, 50, 131072, 3])
        print(sequences_gt_poses.shape) # torch.Size([2, 50, 4, 4])
        print(labels) 

        # -- transform the ground truth poses to the lidar frame
        for i in range(len(labels)): 
            T_GT_L = gt_poses_trans[i]
            original_gt_poses = sequences_gt_poses[i]
            for j in range(sequences_gt_poses[i].shape[0]):
                sequences_gt_poses[i][j] = sequences_gt_poses[i][j] @ T_GT_L 

            # # translation visualization
            # translations = [pose[:3, 3].numpy() for pose in sequences_gt_poses[i]]
            # points = np.array(translations)
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(points)
            # o3d.visualization.draw_geometries([point_cloud])


        # -- reinitialize the neural points
        for index, label in enumerate(labels):
            slamdataset = slamdataset_list[label]
            
            if idx_batch > 0:
                slamdataset.clear()
                neural_points_list[label].clear()
                mapper_list[label] = Mapper(config, slamdataset, neural_points_list[label], geo_mlp, None, None)
            slamdataset.gt_poses = sequences_gt_poses[index]
                       
        # -- train the pc
        for i in range(sequences_pc.shape[1]):
            sequences_pc_curframe = sequences_pc[:, i, :, :]
            sequences_gt_poses_curframe = sequences_gt_poses[:, i, :, :]

            train_multi_sequences(sequences_pc_curframe, sequences_gt_poses_curframe, labels, slamdataset_list, neural_points_list, mapper_list, mesher_list, config, geo_mlp, o3d_vis)
            # print_mlp_weights('geo_mlp weight outside fun', geo_mlp)


        # first_train = False
        torch.save(geo_mlp.state_dict(), os.path.join(checkpoint_dir, f"mlp_decoder_checkpoint{idx_batch}.pth"))  
        idx_batch += 1  

    # Save the model
    torch.save(geo_mlp.state_dict(), os.path.join(checkpoint_dir, 'geo_mlp.pth'))
    print("Model saved to geo_mlp.pth")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()