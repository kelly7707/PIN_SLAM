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

from Dataset_Collection import PointCloudDataset, SingleBagBatchSampler, MultiBagBatchSampler
from dataset.slam_dataset import crop_frame
from dataset.slam_dataset import SLAMDataset as OriginalSLAMDataset
from utils.mesher import Mesher
from utils.tools import voxel_down_sample_torch, setup_wandb, split_chunks
from utils.mapper import Mapper
from utils.visualizer import MapVisualizer

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
    if config.mesh_freq_frame > 0:
        if o3d_vis.render_mesh and (slamdataset.processed_frame == 0 or (slamdataset.processed_frame+1) % config.mesh_freq_frame == 0):     #  or frame_id == last_frame         
            
            # update map bbx
            global_neural_pcd_down = neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=23) # prime number
            slamdataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()
            
            mesh_path = None # no need to save the mesh

            # figure out how to do it efficiently
            if config.mc_local or (not o3d_vis.vis_global): # only build the local mesh
                # cur_mesh = mesher.recon_aabb_mesh(slamdataset.cur_bbx, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)
                chunks_aabb = split_chunks(global_neural_pcd_down, slamdataset.cur_bbx, o3d_vis.mc_res_m*100) # reconstruct in chunks
                cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
            else:
                aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
                chunks_aabb = split_chunks(global_neural_pcd_down, aabb, o3d_vis.mc_res_m*300) # reconstruct in chunks
                cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
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
            

def train_single_sequence(sequence_pc, sequence_gt_poses, slamdataset, neural_points, mapper, mesher, config, geo_mlp, o3d_vis):

    # reinitialize the neural points
    slamdataset.clear()
    neural_points.clear()
    slamdataset.gt_poses = sequence_gt_poses

    for point_cloud, cur_pose_torch in zip(sequence_pc, sequence_gt_poses):
        # point_cloud.to(device=device, dtype=dtype) # nope ; point_cloud = point_cloud.to(device=device, dtype=dtype) #yes
        # cur_pose_torch.to(device=device) # TODO, dtype=dtype ? 
        slamdataset.dataset_process(point_cloud, cur_pose_torch)
        neural_points.travel_dist = torch.tensor(np.array(slamdataset.travel_dist), device = config.device, dtype=config.dtype) 

        cur_sem_labels_torch = None
        mapper.process_frame(slamdataset.cur_point_cloud_torch, cur_sem_labels_torch,
                                cur_pose_torch.detach().to(device=config.device), slamdataset.processed_frame, use_travel_dist= True) # TODO: ignore 'use_travel_dist' -> neural_points.update 'use_travel_dist' temporarily
        # --- for the first frame, we need more iterations to do the initialization (warm-up)
        # cur_iter_num = int(config.iters * config.init_iter_ratio) if first_train else config.iters # 15
        cur_iter_num = config.iters # 15
        first_train = False

        # mapping with fixed poses (every frame)
        mapper.mapping(cur_iter_num) # TODO: neural_points.query_feature -> radius_neighborhood_search (time_filtering should be true)

        if config.o3d_vis_on:
            pin_slam_visual(config, geo_mlp, o3d_vis, slamdataset, neural_points, mapper, mesher)

        if config.wandb_vis_on:
            wandb_log_content = {'frame': slamdataset.processed_frame}
            wandb.log(wandb_log_content)

        slamdataset.processed_frame += 1


def main():
    # --------------------------------------------
    
    # -------------- Dataset Collection ----------
    # New college dataset
    NC_point_cloud_topic = "/os_cloud_node/points"
    nce_bag_path = 'data/Newer_College_Dataset/2021-07-01-10-37-38-quad-easy.bag'
    ncm_bag_path = 'data/Newer_College_Dataset/medium/2021-07-01-11-31-35_0-quad-medium.bag'
    nce_gt_pose_file = 'data/Newer_College_Dataset/gt-nc-quad-easy_TMU.csv'
    ncm_gt_pose_file = 'data/Newer_College_Dataset/medium/gt-nc-quad-medium.csv'

    # ASL
    ASL_point_cloud_topic = "/ouster/points"
    field_bag_path = './data/ASL/field_s/2023-08-09-19-05-05-field_s.bag'
    katzensee_bag_path = 'data/ASL/katzensee/2023-08-21-10-20-22-katzensee_s.bag'
    katzensees_gt_pose_file = 'data/ASL/katzensee/gt-katzensee_s.csv'
    fields_gt_pose_file = 'data/ASL/field_s/gt-field_s.csv'

    # Kitti
    kitti_point_cloud_topic = "/kitti/velo/pointcloud"
    kitti_bag_path = 'data/kitti_example/sequences/00/bag00.bag'

    # list
    # ts_field_name = "t"
    # sequence_paths = [f'{field_bag_path}', f'{katzensee_bag_path}', f'{nce_bag_path}', f'{ncm_bag_path}']
    # point_cloud_topics = [ASL_point_cloud_topic, ASL_point_cloud_topic, NC_point_cloud_topic, NC_point_cloud_topic]
    # gt_poses_files = [fields_gt_pose_file, katzensees_gt_pose_file, nce_gt_pose_file, ncm_gt_pose_file]
    ts_field_name = "t"
    sequence_paths = [f'{nce_bag_path}', f'{ncm_bag_path}']
    point_cloud_topics = [NC_point_cloud_topic, NC_point_cloud_topic]
    gt_poses_files = [nce_gt_pose_file, ncm_gt_pose_file]


    # Create dataset with sequences from all bags
    dataset = PointCloudDataset(sequence_paths, gt_poses_files, point_cloud_topics, num_sequences_per_bag=4) # 8/10
    num_datasets = dataset.unique_bag_labels.shape[0]
    # Use custom batch sampler for one or two sequences from the same bag per batch
    # # batch_sampler = SingleBagBatchSampler(dataset, sequences_per_batch=2)
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
        wandb.run.name = "testing0 |" + run_name
        # Set a description for the run
        wandb.run.notes = "nce+ ncm - 2*5s sequences -- 2 sequences per batch"

    
    geo_mlp = Attention(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)

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


    # --------------------------------------------
    first_train = True
    for batch in dataloader:
        """Each batch contains sequences from all dataset (one sequence per dataset), train them in parallel.
        contains three components:
            1. Point cloud sequences (list of tensors) 
            2. Ground truth poses (list of tensors)
            3. Labels (list of bag file paths)"""

        # Unpacking the batch into sequences and labels
        sequences_pc, sequences_gt_poses, labels = batch

        processes = []
        i = 0
        for sequence_pc, sequence_gt_pose, label in zip(sequences_pc, sequences_gt_poses, labels):
            print(f"Sequence {i+1} shape: {sequence_pc.shape}")  # torch.Size([N, 3])
            print(f"Sequence {i+1} gt poses: {sequences_gt_poses[i].shape}")
            print(f"Label {i+1}: {labels[i]}") 
            i += 1

            p = mp.Process(target=train_single_sequence, args=(sequence_pc, sequence_gt_pose, slamdataset_list[label], neural_points_list[label], mapper_list[label], mesher_list[label], config, geo_mlp, o3d_vis)) 
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        first_train = False
            

    # Save the model
    torch.save(geo_mlp.state_dict(), 'geo_mlp.pth')
    print("Model saved to geo_mlp.pth")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()