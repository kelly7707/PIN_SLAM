#!/usr/bin/env python3
# @file      slam_dataset.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import os
import sys
import numpy as np
from numpy.linalg import inv
import math
import pykitti.utils
import torch
from torch.utils.data import Dataset
import contextlib
import open3d as o3d
from tqdm import tqdm
from rich import print
import csv
from typing import List
import matplotlib.cm as cm
import wandb
import pypose as pp
from kitti360scripts.devkits.commons import loadCalibration
import yaml


from utils.config import Config
from utils.tools import get_time, voxel_down_sample_torch, deskewing,deskewing_IMU, transform_torch, plot_timing_detail, tranmat_close_to_identity
from utils.semantic_kitti_utils import *
from eval.eval_traj_utils import *
# from utils.pgo import PoseGraphManager

import pykitti
import datetime as dt
from scipy.spatial.transform import Rotation as R
# TODO: write a new dataloader for RGB-D inputs, not always firstly converting them to KITTI Lidar format

# ros related
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

class SLAMDataset(Dataset):
    def __init__(self, config: Config) -> None:

        super().__init__()

        self.config = config
        self.silence = config.silence
        self.dtype = config.dtype
        self.device = config.device

        # point cloud files
        if config.pc_path != "": # default empty in ros
            from natsort import natsorted 
            # self.pc_filenames = natsorted(os.listdir(config.pc_path)) # sort files as 1, 2,… 9, 10 not 1, 10, 100 with natsort
            self.pc_filenames = natsorted(os.listdir(os.path.join(config.pc_path,'data')))
            self.total_pc_count = len(self.pc_filenames)
        # if config.sync_imu_path != "":
        #     from natsort import natsorted 
        #     self.sync_imu_filenames = natsorted(os.listdir(config.sync_imu_path)) # sort files as 1, 2,… 9, 10 not 1, 10, 100 with natsort
        #     self.total_sync_imu_count = len(self.sync_imu_filenames)
        
        # pose related
        self.gt_pose_provided = True
        if config.pose_path == '':
            self.gt_pose_provided = False # default

        self.odom_poses = None
        if config.track_on:
            self.odom_poses = []
            
        self.pgo_poses = None
        if config.pgo_on or config.imu_pgo:
            self.pgo_poses = []

        self.gt_poses = None
        if self.gt_pose_provided:
            self.gt_poses = []

        self.poses_w = None
        self.poses_w_closed = None

        self.travel_dist = []

        self.time_table = []

        self.poses_ref = [np.eye(4)] # only used when gt_pose_provided

        # TODO: config-calibration from kitti
        # calibration_kitti360 = False
        if not self.config.source_from_ros:
            # calibration kitti 360
            self.calib360 = {}
            fileCameraToPose = os.path.join(config.calib360_path, 'calib_cam_to_pose.txt')
            T_cam_to_pose = loadCalibration.loadCalibrationCameraToPose(fileCameraToPose) # cam to imu
            # print('Loaded %s' % fileCameraToPose)
            # print('----------cam to Pose--------',T_cam_to_pose)

            fileCameraToVelo = os.path.join(config.calib360_path, 'calib_cam_to_velo.txt')
            T_cam_to_velo = loadCalibration.loadCalibrationRigid(fileCameraToVelo)
            # print('----------cam to Lidar--------',T_cam_to_velo)

            # self.T_pose_to_velo = T_cam_to_pose['image_00'] @ np.linalg.inv(T_cam_to_velo) #from imu to cam @ from cam to lidar
            self.T_L_I = T_cam_to_velo @ np.linalg.inv(T_cam_to_pose['image_00']) 

            self.T_L_I[:3,:3] = np.eye(3)
            self.T_I_L = np.linalg.inv(self.T_L_I)
        
        # TODO config
        # calibration_newer_college = True
        if self.config.source_from_ros:
            calib_file_path = 'data/Newer_College_Dataset/os_imu_lidar_transforms.yaml'
            with open(calib_file_path, 'r') as file:
                calibration_data = yaml.safe_load(file)
            
            # Extract translation and rotation for os_sensor_to_os_imu
            os_sensor_to_os_imu_data = calibration_data['os_sensor_to_os_imu']
            translation_imu = np.array(os_sensor_to_os_imu_data['translation'])
            rotation_imu = np.array(os_sensor_to_os_imu_data['rotation'])
            
            self.T_I_L = create_homogeneous_transform(translation_imu, rotation_imu)
            self.T_L_I = np.linalg.inv(self.T_I_L)
            # -- test: Extract translation and rotation for os_imu_to_os_sensor
            os_imu_to_os_sensor_data = calibration_data['os_imu_to_os_sensor']
            translation_sensor = np.array(os_imu_to_os_sensor_data['translation'])
            rotation_sensor = np.array(os_imu_to_os_sensor_data['rotation'])

            os_imu_to_os_sensor_matrix = create_homogeneous_transform(translation_sensor, rotation_sensor)
            assert np.allclose(np.linalg.inv(self.T_I_L), os_imu_to_os_sensor_matrix)

        self.calib = {}
        self.calib['Tr'] = np.eye(4) # as default if calib file is not provided # as T_lidar<-camera
        
        if self.gt_pose_provided: # default false
            if config.calib_path != '':
                self.calib = read_kitti_format_calib(config.calib_path)
            # TODO: this should be updated, select the pose with correct format, tum format may not endwith csv
            if config.pose_path.endswith('txt'):
                poses_uncalib = read_kitti_format_poses(config.pose_path)
                if config.closed_pose_path is not None:
                    poses_closed_uncalib = read_kitti_format_poses(config.closed_pose_path)
            elif config.pose_path.endswith('csv'):
                poses_uncalib = read_tum_format_poses_csv(config.pose_path)
            else: 
                sys.exit("Wrong pose file format. Please use either *.txt or *.csv")

            # apply calibration
            # actually from camera frame to LiDAR frame, lidar pose in world frame 
            self.poses_w = apply_kitti_format_calib(poses_uncalib, inv(self.calib['Tr'])) 
            if config.closed_pose_path is not None:
                self.poses_w_closed = apply_kitti_format_calib(poses_closed_uncalib, inv(self.calib['Tr'])) 

            # pose in the reference frame (might be the first frame used)
            self.poses_ref = self.poses_w  # initialize size

            if len(self.poses_w) != self.total_pc_count:
                sys.exit("Number of the pose and point cloud are not identical")

            # get the pose in the reference frame
            begin_flag = False
            begin_pose_inv = np.eye(4)
            
            for frame_id in range(self.total_pc_count):
                if not begin_flag:  # the first frame used
                    begin_flag = True
                    if config.first_frame_ref: # use the first frame as the reference (identity)
                        begin_pose_inv = inv(self.poses_w[frame_id])  # T_rw      
                self.poses_ref[frame_id] = begin_pose_inv @ self.poses_w[frame_id]
        # or we directly use the world frame as reference
 
        self.processed_frame: int = 0
        self.shift_ts: float = 0.0

        self.lose_track: bool = False # the odometry lose track or not (for robustness)
        self.consecutive_lose_track_frame: int = 0

        self.color_available: bool = False

        self.intensity_available: bool = False
        
        self.color_scale: float = 255.
        
        self.T_Wl_Llast = np.eye(4)
        self.last_odom_tran = np.eye(4)
        self.T_Wl_Lcur = np.eye(4)
        if self.config.kitti_correction_on:
            self.last_odom_tran[0,3] = self.config.max_range*1e-2 # inital guess for booting on x aixs
            self.color_scale = 1.

        # count the consecutive stop frame of the robot
        self.stop_count: int = 0
        self.stop_status = False

        # current frame point cloud (for visualization)
        self.cur_frame_o3d = o3d.geometry.PointCloud()
        # current frame bounding box in the world coordinate system
        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()
        # merged downsampled point cloud (for visualization)
        self.map_down_o3d = o3d.geometry.PointCloud()
        # map bounding box in the world coordinate system
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()

        self.static_mask = None

        # current frame's data
        self.cur_point_cloud_torch = None
        self.cur_point_ts_torch = None
        self.cur_sem_labels_torch = None
        self.cur_sem_labels_full = None

        # source data for registration
        self.cur_source_points = None
        self.cur_source_normals = None
        self.cur_source_colors = None

        if not self.config.source_from_ros: # kitti360
            # ts # default empty in ros
            self.ts_pc = loadTimestamps(self.config.pc_path)
            self.ts_syncimu = loadTimestamps(self.config.sync_imu_path)
            self.ts_rawimu = loadTimestamps(self.config.raw_imu_path)
        else:
            self.lidar_frame_ts = {}

        # testing IMU
        self.vidual_poses = []
        self.visual_poses_direction = []
        self.visual_lidar_poses = []
        self.visual_lidar_poses_direction = []
        self.visual_imu_frame=[]
        self.visual_imu_frame_direction=[]
        self.vidual_gtposes = []

        # testing tracking
        # pose: imu preintegration input(last frame)/output; estimated by tracking (initial value); optimized value
        # velocity: imu preintegration input(last frame = optimized last frame)/output; optimized
        

    def read_frame_ros(self, lidar_msg, ts_field_name = "time", ts_col=3):
        # ts_col represents the column id for timestamp
        self.T_Wl_Lcur = np.eye(4)
        self.cur_pose_torch = torch.tensor(self.T_Wl_Lcur, device=self.device, dtype=self.dtype)

        pc_data = point_cloud2.read_points(lidar_msg, field_names=("x", "y", "z", ts_field_name), skip_nans=True)
        # pc_data = point_cloud2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
        # convert the point cloud data to a numpy array
        data = np.array(list(pc_data))

        # print(data)

        # how to read the timestamp information
        if ts_col > data.shape[1]-1:
            point_ts = None
        else: # TO-DO: check 
            point_ts = data[:, ts_col]

            if self.processed_frame == 0:
                self.shift_ts = point_ts[0]

            point_ts = point_ts - self.shift_ts
            # point_ts = np.vectorize(dt.timedelta)(microseconds=point_ts / 1000).tolist()
            # TO-DO: nanosec -> will be normalized to 0-1

        # print(point_ts)
        
        point_cloud = data[:,:3]
        
        if point_ts is None:
            print("The point cloud message does not contain the time stamp field:", ts_field_name)

        self.cur_point_cloud_torch = torch.tensor(point_cloud, device=self.device, dtype=self.dtype)

        if self.config.deskew:
            self.get_point_ts(point_ts)
           

    def read_frame(self, frame_id):
        
        # load gt pose if available
        if self.gt_pose_provided: # default false
            self.T_Wl_Lcur = self.poses_ref[frame_id]
            self.gt_poses.append(self.T_Wl_Lcur)
        else: # or initialize with identity
            self.T_Wl_Lcur = np.eye(4) # default
        self.cur_pose_torch = torch.tensor(self.T_Wl_Lcur, device=self.device, dtype=self.dtype)

        point_ts = None

        # load point cloud (support *pcd, *ply and kitti *bin format)
        frame_filename = os.path.join(self.config.pc_path, 'data', self.pc_filenames[frame_id])
        if not self.silence:
            print(frame_filename)
        if not self.config.semantic_on: # default
            point_cloud, point_ts = read_point_cloud(frame_filename, self.config.color_channel) #  [N, 3], [N, 4] or [N, 6], may contain color or intensity 
            if self.config.color_channel > 0: # 0
                point_cloud[:,-self.config.color_channel:]/=self.color_scale
            self.cur_sem_labels_torch = None
        else:
            label_filename = os.path.join(self.config.label_path, self.pc_filenames[frame_id].replace('bin','label'))
            point_cloud, sem_labels, sem_labels_reduced = read_semantic_point_label(frame_filename, label_filename) # [N, 4] , [N], [N]
            self.cur_sem_labels_torch = torch.tensor(sem_labels_reduced, device=self.device, dtype=torch.long) # reduced labels (20 classes)
            self.cur_sem_labels_full = torch.tensor(sem_labels, device=self.device, dtype=torch.long) # full labels (>20 classes)
        
        self.cur_point_cloud_torch = torch.tensor(point_cloud, device=self.device, dtype=self.dtype)
        
        if frame_id>0:
            # IMU--1 load IMU(kitti)
            sync_imu_filename = os.path.join(self.config.sync_imu_path, 'data', '%010d.txt'%frame_id)
            
            start_index = find_closest_timestamp_index(self.ts_pc[frame_id-1],self.ts_rawimu)
            end_index = find_closest_timestamp_index(self.ts_pc[frame_id],self.ts_rawimu)
            
            # if frame_id> 100: start_idx = 1085; >0: 41
            # print('------------testing: index of turning point-----------',start_index)
            self.read_raw_imu(start_index, end_index)
            # frame_id * 10 

        if self.config.deskew:
            self.get_point_ts(point_ts)
        
        # print(self.cur_point_ts_torch)
    
    def read_raw_imu(self, start_index, end_index):
        '''load raw imu data & ts [start_index, end_index-1], source pypose/imu_dataset.py'''
        # all_imu_data = []
        all_imu_filenames =[]
        for idx in range(start_index, end_index):# the last one not included (the start of next frame)
            raw_imu_filename = os.path.join(self.config.raw_imu_path, 'data', f'{idx:010d}.txt')
            all_imu_filenames.append(raw_imu_filename)
            # cur_raw_imu = np.loadtxt(raw_imu_filename)
            # all_imu_data.append(cur_raw_imu)
        
        self.oxts_raw_imu_curinterval = pykitti.utils.load_oxts_packets_and_poses(all_imu_filenames)
        self.ts_raw_imu_curinterval = self.ts_rawimu[start_index:end_index+1] #[frame_i, frame_i+1]
        self.seq_len = len(self.ts_raw_imu_curinterval) - 1

        self.imu_curinter = {}
        self.imu_curinter['dt'] = np.array([dt.datetime.timestamp(self.ts_raw_imu_curinterval[i+1]) -
                                dt.datetime.timestamp(self.ts_raw_imu_curinterval[i])
                                    for i in range(self.seq_len)]) # torch.size([n,1])
        self.imu_curinter['gyro'] = np.array([[self.oxts_raw_imu_curinterval[i].packet.wx,
                                   self.oxts_raw_imu_curinterval[i].packet.wy,
                                   self.oxts_raw_imu_curinterval[i].packet.wz]
                                   for i in range(self.seq_len)]) # torch.size([n,3])
        self.imu_curinter['acc'] = np.array([[self.oxts_raw_imu_curinterval[i].packet.ax,
                                  self.oxts_raw_imu_curinterval[i].packet.ay,
                                  self.oxts_raw_imu_curinterval[i].packet.az]
                                  for i in range(self.seq_len)])
        # # TODO: understand the gt... and Tw_imu?
        # self.imu_curinter['gt_rot'] = pp.euler2SO3(torch.tensor([[self.oxts_raw_imu_curinterval[i].packet.roll,
        #                                           self.oxts_raw_imu_curinterval[i].packet.pitch,
        #                                           self.oxts_raw_imu_curinterval[i].packet.yaw]
        #                                           for i in range(self.seq_len)]))
        # self.imu_curinter['gt_vel'] = self.imu_curinter['gt_rot'] @ torch.tensor([[self.oxts_raw_imu_curinterval[i].packet.vf,
        #                                            self.oxts_raw_imu_curinterval[i].packet.vl,
        #                                            self.oxts_raw_imu_curinterval[i].packet.vu]
        #                                            for i in range(self.seq_len)])
        # print('-------------------', self.imu_curinter['gt_vel'][:1])
        # self.imu_curinter['gt_pos'] = torch.tensor(np.array([self.oxts_raw_imu_curinterval[i].T_w_imu[0:3, 3]
        #                                      for i in range(self.seq_len)]))
        
        # -- visual data & bias
        visual_data_bias = False
        if visual_data_bias:
            if start_index >=35:
                gyro_data = self.imu_curinter['gyro']
                acc_data = self.imu_curinter['acc']
                timestamps = np.cumsum(self.imu_curinter['dt'])  # Accumulating the time differences to get the timestamps

                gyroBias = np.mean(self.imu_curinter['gyro'], axis=0)
                accBias = np.mean(self.imu_curinter['acc'], axis=0) - np.array([0, 0, 9.8])
                gyro_sigma = (gyro_data.max(axis=0) - gyro_data.min(axis=0)) / 2
                acc_sigma = (acc_data.max(axis=0) - acc_data.min(axis=0)) / 2
                # Plotting the IMU data
                fig, axs = plt.subplots(2, 1, figsize=(10, 8))
                # Gyroscope data
                axs[0].plot(timestamps, gyro_data[:, 0], label='Gyro X', color='r')
                axs[0].plot(timestamps, gyro_data[:, 1], label='Gyro Y', color='g')
                axs[0].plot(timestamps, gyro_data[:, 2], label='Gyro Z', color='b')
                axs[0].axhline(gyroBias[0], color='r', linestyle='--', label='Bias X')
                axs[0].axhline(gyroBias[1], color='g', linestyle='--', label='Bias Y')
                axs[0].axhline(gyroBias[2], color='b', linestyle='--', label='Bias Z')
                axs[0].set_title('Gyroscope Data')
                axs[0].set_xlabel('Time [s]')
                axs[0].set_ylabel('Angular Velocity [rad/s]')
                axs[0].legend()
                axs[0].grid(True)
                # Display gyro sigmas on the plot
                axs[0].text(0.05, 0.95, f'Gyro Bias X: {gyroBias[0]:.6f}\nGyro Bias Y: {gyroBias[1]:.6f}\nGyro Bias Z: {gyroBias[2]:.6f}\n'
                                        f'Gyro Sigma X: {gyro_sigma[0]:.6f}\nGyro Sigma Y: {gyro_sigma[1]:.6f}\nGyro Sigma Z: {gyro_sigma[2]:.6f}',
                            transform=axs[0].transAxes, fontsize=10, verticalalignment='top')

                # Accelerometer data
                axs[1].plot(timestamps, acc_data[:, 0], label='Accel X', color='r')
                axs[1].plot(timestamps, acc_data[:, 1], label='Accel Y', color='g')
                axs[1].plot(timestamps, acc_data[:, 2], label='Accel Z', color='b')
                axs[1].axhline(accBias[0], color='r', linestyle='--', label='Bias X + Gravity')
                axs[1].axhline(accBias[1], color='g', linestyle='--', label='Bias Y')
                axs[1].axhline(accBias[2] + 9.8, color='b', linestyle='--', label='Bias Z')
                axs[1].set_title('Accelerometer Data')
                axs[1].set_xlabel('Time [s]')
                axs[1].set_ylabel('Linear Acceleration [m/s^2]')
                axs[1].legend()
                axs[1].grid(True)
                # Display accel sigmas on the plot
                axs[1].text(0.05, 0.95, f'Accel Bias X: {accBias[0]:.6f}\nAccel Bias Y: {accBias[1]:.6f}\nAccel Bias Z: {accBias[2]:.6f}\n'
                                        f'Accel Sigma X: {acc_sigma[0]:.6f}\nAccel Sigma Y: {acc_sigma[1]:.6f}\nAccel Sigma Z: {acc_sigma[2]:.6f}',
                            transform=axs[1].transAxes, fontsize=10, verticalalignment='top')


                plt.tight_layout()
                plt.show()


    
    # point-wise timestamp is now only used for motion undistortion (deskewing)
    def get_point_ts(self, point_ts = None):
        if self.config.deskew:
            if point_ts is not None and self.config.valid_ts_in_points: # ros default
                self.cur_point_ts_torch = torch.tensor(point_ts, device=self.device, dtype=self.dtype)
            else: # default
                H = 64
                W = 1024
                if self.cur_point_cloud_torch.shape[0] == H*W:  # for Ouster 64-beam LiDAR
                    if not self.silence:
                        print("Ouster-64 point cloud deskewed")
                    self.cur_point_ts_torch = (torch.floor(torch.arange(H * W) / H) / W).reshape(-1, 1).to(self.cur_point_cloud_torch)
                else: # default
                    yaw = -torch.atan2(self.cur_point_cloud_torch[:,1], self.cur_point_cloud_torch[:,0])  # y, x -> rad (clockwise)
                    if self.config.lidar_type_guess == "velodyne": # default
                        # for velodyne LiDAR (from -x axis, clockwise)
                        self.cur_point_ts_torch = 0.5 * (yaw / math.pi + 1.0) # [0,1]; yaw / math.pi-> [-1,1]
                        if not self.silence:
                            print("Velodyne point cloud deskewed")
                    else:
                        # for Hesai LiDAR (from +y axis, clockwise)
                        self.cur_point_ts_torch = 0.5 * (yaw / math.pi + 0.5) # [-0.25,0.75]
                        self.cur_point_ts_torch[self.cur_point_ts_torch < 0] += 1.0 # [0,1]
                        if not self.silence:
                            print("HESAI point cloud deskewed")


    def preprocess_frame(self, pgm, frame_id=0): #: PoseGraphManager

        T1 = get_time()

        if self.config.adaptive_range_on:
            pc_max_bound, _ = torch.max(self.cur_point_cloud_torch[:, :3], dim=0)
            pc_min_bound, _ = torch.min(self.cur_point_cloud_torch[:, :3], dim=0)

            min_x_range = min(torch.abs(pc_max_bound[0]),  torch.abs(pc_min_bound[0]))
            min_y_range = min(torch.abs(pc_max_bound[1]),  torch.abs(pc_min_bound[1]))
            max_x_y_min_range = max(min_x_range, min_y_range)

            crop_max_range = min(self.config.max_range, 2.*max_x_y_min_range)
        else:
            crop_max_range = self.config.max_range
        
        # adaptive
        train_voxel_m = (crop_max_range/self.config.max_range) * self.config.vox_down_m
        source_voxel_m = (crop_max_range/self.config.max_range) * self.config.source_vox_down_m

        # down sampling (together with the color and semantic entities)
        original_count = self.cur_point_cloud_torch.shape[0]
        if self.config.rand_downsample:
            kept_count = int(original_count*self.config.rand_down_r)
            idx = torch.randint(0, original_count, (kept_count,), device=self.device)
        else:
            idx = voxel_down_sample_torch(self.cur_point_cloud_torch[:,:3], train_voxel_m)
        self.cur_point_cloud_torch = self.cur_point_cloud_torch[idx]
        if self.cur_point_ts_torch is not None:
            self.cur_point_ts_torch = self.cur_point_ts_torch[idx]
        if self.cur_sem_labels_torch is not None:
            self.cur_sem_labels_torch = self.cur_sem_labels_torch[idx]
            self.cur_sem_labels_full = self.cur_sem_labels_full[idx]
        
        T2 = get_time()

        # preprocessing, filtering
        if self.cur_sem_labels_torch is not None:
            self.cur_point_cloud_torch, self.cur_sem_labels_torch = filter_sem_kitti(self.cur_point_cloud_torch, self.cur_sem_labels_torch, self.cur_sem_labels_full,
                                                                                     True, self.config.filter_moving_object) 
        else:
            self.cur_point_cloud_torch, self.cur_point_ts_torch = crop_frame(self.cur_point_cloud_torch, self.cur_point_ts_torch, 
                                                                             self.config.min_z, self.config.max_z, 
                                                                             self.config.min_range, crop_max_range)

        if self.config.kitti_correction_on:
            self.cur_point_cloud_torch = intrinsic_correct(self.cur_point_cloud_torch, self.config.correction_deg) # TODO

        T3 = get_time()

        init_imu_integrator = {}
        # prepare for the registration
        if self.processed_frame == 0: # initialize the first frame, no tracking yet
            # self.T_Wl_Wi=  np.linalg.inv( pgm.T_Wi_I0 @ self.T_I_L) # wi to wl
            self.T_Wl_Wi= self.T_L_I @ np.linalg.inv( pgm.T_Wi_I0 ) # wi to wl
            
            #
            if self.config.track_on:
                self.odom_poses.append(self.T_Wl_Lcur)
            if self.config.pgo_on or self.config.imu_pgo:
                self.pgo_poses.append(self.T_Wl_Lcur)   
            if self.gt_pose_provided and frame_id > 0: # not start with the first frame, default false
                self.last_odom_tran = inv(self.poses_ref[frame_id-1]) @ self.T_Wl_Lcur # T_last<-cur
            self.travel_dist.append(0.)
            self.T_Wl_Llast = self.T_Wl_Lcur

            # # Lidar test
            self.estimated_pose_lidar = self.T_Wl_Lcur
            # # IMU test - frame0 initialization
            self.imu_preinte_continuous_test = self.T_Wl_Llast

        elif self.processed_frame > 0: 
            # # pose initial guess
            # # original - PIN SLAM
            # last_tran = np.linalg.norm(self.last_odom_tran[:3,3])  # from update_odom_pose()
            # # if self.config.uniform_motion_on and not self.lose_track and last_tran > 0.2 * self.config.voxel_size_m: # apply uniform motion model here
            # if self.config.uniform_motion_on and not self.lose_track: # default: apply uniform motion model here
            #     cur_pose_init_guess = self.T_Wl_Llast @ self.last_odom_tran # T_world<-cur = T_world<-last @ T_last<-cur
            # else: # static initial guess
            #     cur_pose_init_guess = self.T_Wl_Llast

            if not self.config.track_on and self.gt_pose_provided: # default off
                cur_pose_init_guess = self.poses_ref[frame_id]

            
            # # test imu preintegration (world coord: east-north-up / x = forward, y = left, z = up)
            # # IMU- TEST preintegration
            # # last_pose_w2imu = self.imu_preinte_continuous_test
            # last_pose_w2imu = np.linalg.inv(self.T_Wl_Wi) @ self.imu_preinte_continuous_test @ self.T_L_I 
            # T_Wi_I = pgm.preintegration(acc=self.imu_curinter['acc'],gyro=self.imu_curinter['gyro'],dts=self.imu_curinter['dt'],last_pose=last_pose_w2imu, cur_id=self.processed_frame)
            # T_Wl_Lcur = self.T_Wl_Wi @ T_Wi_I @ self.T_I_L
            # # T_Wl_Lcur = T_Wi_I
            # self.imu_preinte_continuous_test = T_Wl_Lcur
            # T_Wl_I = self.T_Wl_Wi @ T_Wi_I


            # # IMU--2 pose initial guess
            # # transform to imu frame
            T_Wi_Ilast = np.linalg.inv(self.T_Wl_Wi) @ self.T_Wl_Llast @ self.T_L_I 
            if frame_id==1: 
                assert np.allclose(T_Wi_Ilast, pgm.T_Wi_I0)
            # last_pose_w2imu = self.imu_preinte_continuous_test @ self.T_pose_to_velo
            T_Wi_Icur = pgm.preintegration(acc=self.imu_curinter['acc'],gyro=self.imu_curinter['gyro'],dts=self.imu_curinter['dt'],last_pose=T_Wi_Ilast, cur_id=self.processed_frame)
            T_Wl_Lcur = self.T_Wl_Wi @ T_Wi_Icur @ self.T_I_L
            T_Wl_I = self.T_Wl_Wi @ T_Wi_Icur
            T_Llast_Lcur =  np.linalg.inv(self.T_Wl_Llast) @ T_Wl_Lcur
            
            cur_pose_init_guess = T_Wl_Lcur


            # --- testing: IMU preinte vidual
            poses = T_Wl_Lcur  #initial_guess_w2imu # initial_guess_w2lidar # output, wrt. initial imu frame
            self.vidual_poses.append(poses[:3, 3])
            self.visual_poses_direction.append(poses[:3, :3])
            self.visual_lidar_poses.append(self.estimated_pose_lidar[:3, 3])
            self.visual_lidar_poses_direction.append(self.estimated_pose_lidar[:3, :3])
            self.visual_imu_frame.append(T_Wl_I[:3, 3])
            self.visual_imu_frame_direction.append(T_Wl_I[:3, :3])

            
            
            plot = '2d'
            if frame_id % 400==0:
                visual_poses_np = np.array(self.vidual_poses)
                visual_poses_direction_np = np.array(self.visual_poses_direction)
                visual_poses_lidar_np = np.array(self.visual_lidar_poses)
                visual_poses_dirction_lidar_np = np.array(self.visual_lidar_poses_direction)
                visual_imu_frame_np = np.array(self.visual_imu_frame)
                visual_imu_frame_direction_np = np.array(self.visual_imu_frame_direction)
                if plot == '2d':
                    plt.figure(figsize=(5, 5))                

                    # ax = plt.axes(projection='3d')
                    # ax.plot3D(visual_poses_np[:,0], visual_poses_np[:,1], visual_poses_np[:,2], 'b')
                    # # ax.plot3D(visual_gtposes_np[:,0], visual_gtposes_np[:,1], visual_gtposes_np[:,2], 'r')

                    plt.plot(visual_poses_np[0,0], visual_poses_np[0,1], 'bo', markersize=10)
                    plt.plot(visual_poses_np[:,0], visual_poses_np[:,1], 'b')  # Blue line for the trajectory
                    plt.plot(visual_poses_lidar_np[:,0], visual_poses_lidar_np[:,1], 'r')

                    plt.title("Gtsam IMU Integrator")
                    # plt.legend(["PyPose", "Ground Truth"])
                    figure = os.path.join('/home/zjw/master_thesis/visual/testing'+f'gtsam IMU preintegration_{frame_id}'+'.png')
                    plt.savefig(figure)
                    print("Saved to", figure)

                    # # -- 
                    # plt.figure(figsize=(5, 5)) 
                    # plt.plot(visual_poses_lidar_np[:,0], visual_poses_lidar_np[:,1], 'r')
                    # plt.title("Lidar IMU Integrator")
                    # figure = os.path.join('/home/zjw/master_thesis/visual/testing'+f'Lidar IMU preintegration_{frame_id}'+'.png')
                    # plt.savefig(figure)
                else:
                    # Create a figure
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(111, projection='3d')

                    # Plot the poses with orientation vectors
                    for i in range(len(visual_poses_np)):
                        plot_frame(ax, visual_poses_direction_np[i], visual_poses_np[i], 'Pose', 'k', ['r', 'g', 'b'], length=0.05)
                        plot_frame(ax, visual_poses_dirction_lidar_np[i], visual_poses_lidar_np[i], 'Pose', 'k', ['r', 'g', 'b'], length=0.05)
                        plot_frame(ax, visual_imu_frame_direction_np[i], visual_imu_frame_np[i], 'Pose', 'k', ['r', 'g', 'b'], length=0.05)

                    # Optionally, plot the LiDAR poses
                    # ax.plot3D(visual_poses_lidar_np[:, 0], visual_poses_lidar_np[:, 1], visual_poses_lidar_np[:, 2], 'c')

                    # Set labels and title
                    ax.set_xlabel('X-axis')
                    ax.set_ylabel('Y-axis')
                    ax.set_zlabel('Z-axis')
                    plt.title("3D Poses with Orientation Vectors and LiDAR Poses")

                    # Save the figure
                    frame_id = 0  # Replace with your actual frame ID
                    figure_path = os.path.join('/home/zjw/master_thesis/visual/testing', f'gtsam_IMU_preintegration_{frame_id}.png')
                    # plt.savefig(figure_path)
                    print("Saved to", figure_path)

                    # Show the plot
                    plt.show()


            # --- pose initial guess tensor
            self.cur_pose_guess_torch = torch.tensor(cur_pose_init_guess, dtype=torch.float64, device=self.device)   
            cur_source_torch = self.cur_point_cloud_torch.clone() # used for registration
            
            # source point voxel downsampling (for registration)
            idx = voxel_down_sample_torch(cur_source_torch[:,:3], source_voxel_m)
            cur_source_torch = cur_source_torch[idx]
            self.cur_source_points = cur_source_torch[:,:3]
            if self.config.color_on:
                self.cur_source_colors = cur_source_torch[:,3:]
            
            if self.cur_point_ts_torch is not None:
                cur_ts = self.cur_point_ts_torch.clone()
                cur_source_ts = cur_ts[idx]
            else:
                cur_source_ts = None

            # deprecated # key points extraction for registration
            # if self.config.estimate_normal:
            #     source_points = self.cur_source_points.clone()
            #     cur_hasher = VoxelHasherIndex(source_points, source_voxel_m, buffer_size=int(1e6))
            #     neighb_idx = cur_hasher.radius_neighborhood_search(source_points, source_voxel_m)
            #     valid_mask = neighb_idx > 0
            #     neighbors = source_points[neighb_idx] # fix the point corresponding to -1 idx
            #     neighbors[~valid_mask] = torch.ones(3).to(source_points) * 9999.
            #     extractor = GeometricFeatureExtractor()
            #     _, self.cur_source_normals, valid_normal_mask = extractor(source_points, neighbors, source_voxel_m)   
            #     self.cur_source_points = self.cur_source_points[valid_normal_mask]
            #     if self.config.color_on:
            #         self.cur_source_colors = self.cur_source_colors[valid_normal_mask]
            #     if self.cur_point_ts_torch is not None:
            #         cur_source_ts = cur_source_ts[valid_normal_mask]
            # else:
            #     self.cur_source_normals = None

            
            # TODO: not only for frame_id>0
            # deskewing (motion undistortion) for source point cloud
            if self.config.deskew and not self.lose_track:
                # self.cur_source_points = deskewing(self.cur_source_points, cur_source_ts, 
                #                                    torch.tensor(self.last_odom_tran, device=self.device, dtype=self.dtype)) # T_last<-cur
                
                
                # IMU-- 3 deskewing
                T_Wi_Iimu_deskewing = pgm.imu_prediction_poses_curinterval
                # assert T_Wi_Icur == T_Wi_Iimu_deskewing[-1]
                # T_Llast_Limu_deskewing = torch.tensor(self.T_L_I, dtype=torch.float32) @ torch.tensor(np.linalg.inv(T_Wi_Ilast), dtype=torch.float32) @ T_Wi_Iimu_deskewing @ torch.tensor(self.T_I_L, dtype=torch.float32)
                T_Lcur_Limu_deskewing = torch.tensor(np.linalg.inv(T_Wl_Lcur), dtype=torch.float32) @ torch.tensor(self.T_Wl_Wi, dtype=torch.float32) @ T_Wi_Iimu_deskewing @ torch.tensor(self.T_I_L, dtype=torch.float32)
                # assert np.allclose(T_Lcur_Limu_deskewing[-1].numpy(), np.eye(4))  # the transformed poses of last imu prediction should be equals to Identity

                if self.config.source_from_ros:
                    lidar_last_ts = self.lidar_frame_ts['start_ts']
                    lidar_cur_ts = self.lidar_frame_ts['end_ts']
                else:
                    lidar_last_ts = self.ts_pc[frame_id-1]
                    lidar_cur_ts = self.ts_pc[frame_id]
                self.cur_source_points = deskewing_IMU(self.cur_source_points, cur_source_ts, self.ts_raw_imu_curinterval[1:], T_Lcur_Limu_deskewing, np.linalg.inv(T_Llast_Lcur), lidar_last_ts, lidar_cur_ts)
                # test
                # self.cur_point_cloud_torch = self.cur_source_points.clone()
                self.cur_point_cloud_torch = deskewing_IMU(self.cur_point_cloud_torch, self.cur_point_ts_torch, self.ts_raw_imu_curinterval[1:], T_Lcur_Limu_deskewing, np.linalg.inv(T_Llast_Lcur), lidar_last_ts, lidar_cur_ts)
            # print("# Source point for registeration : ", cur_source_torch.shape[0])
    
        T4 = get_time()
    
    def update_odom_pose(self, cur_pose_torch: torch.tensor):
        # needed to be at least the second frame

        self.cur_pose_torch = cur_pose_torch.detach() # need to be out of the computation graph, used for mapping

        self.T_Wl_Lcur = self.cur_pose_torch.cpu().numpy()    
    
        self.last_odom_tran = inv(self.T_Wl_Llast) @ self.T_Wl_Lcur # T_last<-cur

        if tranmat_close_to_identity(self.last_odom_tran, 1e-3, self.config.voxel_size_m*0.1):
            self.stop_count += 1
        else:
            self.stop_count = 0
        
        if self.stop_count > self.config.stop_frame_thre:
            self.stop_status = True
            if not self.silence:
                print("Robot stopped")
        else:
            self.stop_status = False

        if self.config.pgo_on or self.config.imu_pgo: # initialization the pgo pose
            self.pgo_poses.append(self.T_Wl_Lcur) 

        if self.odom_poses is not None:
            cur_odom_pose = self.odom_poses[-1] @ self.last_odom_tran # T_world<-cur
            self.odom_poses.append(cur_odom_pose)

        if len(self.travel_dist) > 0:
            cur_frame_travel_dist = np.linalg.norm(self.last_odom_tran[:3,3])
            if cur_frame_travel_dist > self.config.surface_sample_range_m * 40.0: # too large translation in one frame --> lose track
                self.lose_track = True 
                # sys.exit("Too large translation in one frame, system failed") # FIXME
               
            accu_travel_dist = self.travel_dist[-1] + cur_frame_travel_dist
            self.travel_dist.append(accu_travel_dist)
            if not self.silence:
                print("Accumulated travel distance (m): %f" % accu_travel_dist)
        else: 
            sys.exit("This function needs to be used from at least the second frame")
        
        self.T_Wl_Llast = self.T_Wl_Lcur # update for the next frame
        self.estimated_pose_lidar = self.T_Wl_Lcur

        # deskewing (motion undistortion using the estimated transformation) for the sampled points for mapping
        # if self.config.deskew and not self.lose_track:
        #     self.cur_point_cloud_torch = deskewing(self.cur_point_cloud_torch, self.cur_point_ts_torch, 
        #                                  torch.tensor(self.last_odom_tran, device=self.device, dtype=self.dtype)) # T_last<-cur
        
        if self.lose_track:
            self.consecutive_lose_track_frame += 1
        else:
            self.consecutive_lose_track_frame = 0
        
        # if self.consecutive_lose_track_frame > 20:
            # sys.exit("Lose track for a long time, system failed") # FIXME

    def update_poses_after_pgo(self, pgo_cur_pose, pgo_poses):
        self.T_Wl_Lcur = pgo_cur_pose
        self.T_Wl_Llast = pgo_cur_pose # update for next frame
        self.pgo_poses = pgo_poses # update pgo pose

    def update_o3d_map(self):

        frame_down_torch = self.cur_point_cloud_torch # no futher downsample

        frame_o3d = o3d.geometry.PointCloud()
        frame_points_np = frame_down_torch[:,:3].detach().cpu().numpy().astype(np.float64)
    
        frame_o3d.points = o3d.utility.Vector3dVector(frame_points_np)

        # visualize or not
        # uncomment to visualize the dynamic mask
        if self.config.dynamic_filter_on and self.static_mask is not None:
            static_mask = self.static_mask.detach().cpu().numpy()
            frame_colors_np = np.ones_like(frame_points_np) * 0.7
            frame_colors_np[~static_mask,1:] = 0.0
            frame_colors_np[~static_mask,0] = 1.0
            frame_o3d.colors = o3d.utility.Vector3dVector(frame_colors_np.astype(np.float64))

        frame_o3d = frame_o3d.transform(self.T_Wl_Lcur)

        if self.config.color_channel > 0:
            frame_colors_np = frame_down_torch[:,3:].detach().cpu().numpy().astype(np.float64)
            if self.config.color_channel == 1:
                frame_colors_np = np.repeat(frame_colors_np.reshape(-1, 1),3,axis=1) 
            frame_o3d.colors = o3d.utility.Vector3dVector(frame_colors_np)
        elif self.cur_sem_labels_torch is not None:
            frame_label_torch = self.cur_sem_labels_torch
            frame_label_np = frame_label_torch.detach().cpu().numpy()
            frame_label_color = [sem_kitti_color_map[sem_label] for sem_label in frame_label_np]
            frame_label_color_np = np.asarray(frame_label_color, dtype=np.float64)/255.0
            frame_o3d.colors = o3d.utility.Vector3dVector(frame_label_color_np)

        self.cur_frame_o3d = frame_o3d 
        if self.cur_frame_o3d.has_points():
            self.cur_bbx = self.cur_frame_o3d.get_axis_aligned_bounding_box()

        cur_max_z = self.cur_bbx.get_max_bound()[-1]
        cur_min_z = self.cur_bbx.get_min_bound()[-1]

        bbx_center = self.T_Wl_Lcur[:3,3]
        bbx_min = np.array([bbx_center[0]-self.config.max_range, bbx_center[1]-self.config.max_range, cur_min_z])
        bbx_max = np.array([bbx_center[0]+self.config.max_range, bbx_center[1]+self.config.max_range, cur_max_z])

        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)

        # use the downsampled neural points here (done outside the class)
                                   
    def write_results(self, run_path: str):
        odom_poses_out = apply_kitti_format_calib(self.odom_poses, self.calib['Tr'])
        write_kitti_format_poses(os.path.join(run_path, "odom_poses_"), odom_poses_out)
        write_tum_format_poses(os.path.join(run_path, "odom_poses_"), odom_poses_out)
        write_traj_as_o3d(self.odom_poses, os.path.join(run_path, "odom_poses.ply"))

        if self.config.pgo_on:
            slam_poses_out = apply_kitti_format_calib(self.pgo_poses, self.calib['Tr'])
            write_kitti_format_poses(os.path.join(run_path, "slam_poses_"), slam_poses_out)
            write_tum_format_poses(os.path.join(run_path, "slam_poses_"), slam_poses_out)
            write_traj_as_o3d(self.pgo_poses, os.path.join(run_path, "slam_poses.ply"))
        
        if self.gt_pose_provided:
            write_traj_as_o3d(self.gt_poses, os.path.join(run_path, "gt_poses.ply"))

        time_table = np.array(self.time_table)
        mean_time_s = np.sum(time_table)/self.processed_frame*1.0
        print("Consuming time per frame      (s):", mean_time_s)
        print("Calculated over %d frames" % self.processed_frame)
        np.save(os.path.join(run_path, "time_table.npy"), time_table) # save detailed time table
        plot_timing_detail(time_table, os.path.join(run_path, "time_details.png"), self.config.pgo_on)

        # evaluation report
        if self.gt_pose_provided and len(self.gt_poses) == len(self.odom_poses):
            print("Odometry evaluation:")
            avg_tra, avg_rot = relative_error(self.gt_poses, self.odom_poses) # fix the rotation error issues (done)
            ate_rot, ate_trans, align_mat = absolute_error(self.gt_poses, self.odom_poses, self.config.eval_traj_align)
            print("Average Translation Error     (%):", avg_tra)
            print("Average Rotational Error  (deg/m):", avg_rot)
            print("Absoulte Trajectory Error     (m):", ate_trans)
            print("Absoulte Rotational Error   (deg):", ate_rot)

            if self.config.wandb_vis_on:
                wandb_log_content = {'Average Translation Error [%]': avg_tra, 'Average Rotational Error [deg/m]': avg_rot,
                                     'Absoulte Trajectory Error [m]': ate_trans, 'Absoulte Rotational Error [deg]': ate_rot,
                                     'Consuming time per frame [s]': mean_time_s} 
                wandb.log(wandb_log_content)

            if self.config.pgo_on and len(self.gt_poses) == len(self.pgo_poses):
                print("SLAM evaluation:")
                avg_tra_slam, avg_rot_slam = relative_error(self.gt_poses, self.pgo_poses)
                ate_rot_slam, ate_trans_slam, align_mat_slam = absolute_error(self.gt_poses, self.pgo_poses, self.config.eval_traj_align)
                print("Average Translation Error     (%):", avg_tra_slam)
                print("Average Rotational Error  (deg/m):", avg_rot_slam)
                print("Absoulte Trajectory Error     (m):", ate_trans_slam)
                print("Absoulte Rotational Error   (deg):", ate_rot_slam)

                if self.config.wandb_vis_on:
                    wandb_log_content = {'SLAM Average Translation Error [%]': avg_tra_slam, 'SLAM Average Rotational Error [deg/m]': avg_rot_slam, 'SLAM Absoulte Trajectory Error [m]': ate_trans_slam, 'SLAM Absoulte Rotational Error [deg]': ate_rot_slam} 
                    wandb.log(wandb_log_content)
            
            csv_columns = ['Average Translation Error [%]', 'Average Rotational Error [deg/m]', 'Absoulte Trajectory Error [m]', 'Absoulte Rotational Error [deg]', "Consuming time per frame [s]", "Frame count"]
            pose_eval = [{csv_columns[0]: avg_tra, csv_columns[1]: avg_rot, csv_columns[2]: ate_trans, csv_columns[3]: ate_rot, csv_columns[4]: mean_time_s, csv_columns[5]: int(self.processed_frame)}]
            if self.config.pgo_on:
                slam_eval_dict = {csv_columns[0]: avg_tra_slam, csv_columns[1]: avg_rot_slam, csv_columns[2]: ate_trans_slam, csv_columns[3]: ate_rot_slam, csv_columns[4]: mean_time_s, csv_columns[5]: int(self.processed_frame)}
                pose_eval.append(slam_eval_dict)
            output_csv_path = os.path.join(run_path, "pose_eval.csv")
            try:
                with open(output_csv_path, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in pose_eval:
                        writer.writerow(data)
            except IOError:
                print("I/O error")

            output_traj_plot_path_2d = os.path.join(run_path, "traj_plot_2d.png")
            output_traj_plot_path_3d = os.path.join(run_path, "traj_plot_3d.png")
            # trajectory not aligned yet
            if self.config.pgo_on:
                plot_trajectories(output_traj_plot_path_2d, self.pgo_poses, self.gt_poses, self.odom_poses, plot_3d=False) 
                plot_trajectories(output_traj_plot_path_3d, self.pgo_poses, self.gt_poses, self.odom_poses, plot_3d=True)   
            else:
                plot_trajectories(output_traj_plot_path_2d, self.odom_poses, self.gt_poses, plot_3d=False)
                plot_trajectories(output_traj_plot_path_3d, self.odom_poses, self.gt_poses, plot_3d=True)
    
    def write_merged_point_cloud(self, run_path: str):
        
        print("Begin to replay the dataset ...")

        o3d_device = o3d.core.Device("CPU:0")
        o3d_dtype = o3d.core.float32
        map_out_o3d = o3d.t.geometry.PointCloud(o3d_device)
        map_points_np = np.empty((0, 3))
        map_intensity_np = np.empty(0)
        map_color_np = np.empty((0, 3))

        use_frame_id = 0
        for frame_id in tqdm(range(self.total_pc_count)): # frame id as the idx of the frame in the data folder without skipping
            if (frame_id < self.config.begin_frame or frame_id > self.config.end_frame or frame_id % self.config.every_frame != 0):
                continue

            self.read_frame(frame_id)

            if self.config.kitti_correction_on:
                self.cur_point_cloud_torch = intrinsic_correct(self.cur_point_cloud_torch, self.config.correction_deg)

            if self.config.deskew and use_frame_id < self.processed_frame-1:
                if self.config.track_on:
                    tran_in_frame = np.linalg.inv(self.odom_poses[use_frame_id+1]) @ self.odom_poses[use_frame_id]
                elif self.gt_pose_provided:
                    tran_in_frame = np.linalg.inv(self.gt_poses[use_frame_id+1]) @ self.gt_poses[use_frame_id]
                self.cur_point_cloud_torch = deskewing(self.cur_point_cloud_torch, self.cur_point_ts_torch, 
                                                       torch.tensor(tran_in_frame, device=self.device, dtype=torch.float64)) # T_last<-cur
            
            down_vox_m = self.config.vox_down_m
            idx = voxel_down_sample_torch(self.cur_point_cloud_torch[:,:3], down_vox_m)
            
            frame_down_torch = self.cur_point_cloud_torch[idx]

            frame_down_torch, _ = crop_frame(frame_down_torch, None, self.config.min_z, self.config.max_z, self.config.min_range, self.config.max_range)
                                                       
            if self.config.pgo_on:
                cur_pose_torch = torch.tensor(self.pgo_poses[use_frame_id], device=self.device, dtype=torch.float64)
            elif self.config.track_on:
                cur_pose_torch = torch.tensor(self.odom_poses[use_frame_id], device=self.device, dtype=torch.float64)
            elif self.gt_pose_provided:
                cur_pose_torch = torch.tensor(self.gt_poses[use_frame_id], device=self.device, dtype=torch.float64)
            frame_down_torch[:,:3] = transform_torch(frame_down_torch[:,:3], cur_pose_torch) 

            frame_points_np = frame_down_torch[:,:3].detach().cpu().numpy()
            map_points_np = np.concatenate((map_points_np, frame_points_np), axis=0)
            if self.config.color_channel == 1:
                frame_intensity_np = frame_down_torch[:,3].detach().cpu().numpy()
                map_intensity_np = np.concatenate((map_intensity_np, frame_intensity_np), axis=0)
            elif self.config.color_channel == 3:
                frame_color_np = frame_down_torch[:,3:].detach().cpu().numpy()
                map_color_np = np.concatenate((map_color_np, frame_color_np), axis=0)
            
            use_frame_id += 1
        
        print("Replay done")

        map_out_o3d.point["positions"] =  o3d.core.Tensor(map_points_np, o3d_dtype, o3d_device)
        if self.config.color_channel == 1:
            map_out_o3d.point["intensity"] =  o3d.core.Tensor(np.expand_dims(map_intensity_np, axis=1), o3d_dtype, o3d_device)
        elif self.config.color_channel == 3:
            map_out_o3d.point["colors"] =  o3d.core.Tensor(map_color_np, o3d_dtype, o3d_device)
        
        print("Estimate normal")
        map_out_o3d.estimate_normals(max_nn=20)

        if run_path is not None:
            print("Output merged point cloud map")
            o3d.t.io.write_point_cloud(os.path.join(run_path, "map", "merged_point_cloud.ply"), map_out_o3d)


def find_closest_timestamp_index(target_timestamp, sorted_timestamps):
    # Use searchsorted to find the insertion point
    idx = np.searchsorted(sorted_timestamps, target_timestamp, side='left')
    if idx == len(sorted_timestamps):
        print('------- the last imu --------------')
        if not np.isclose(sorted_timestamps[idx-1].timestamp() * 1e6,target_timestamp.timestamp() * 1e6, atol=0.01):
            print('------- imu wrong here --------------')
    # assert np.isclose(sorted_timestamps[idx].timestamp() * 1e6,target_timestamp.timestamp() * 1e6), 'IMU and LiDAR timestamps are not close'
    # Determine the closest index by checking boundaries
    if idx == 0:
        return 0
        assert 1 == 0
    else:
        # Check if the target timestamp is closer to the current index or the previous index
        before = abs(sorted_timestamps[idx - 1] - target_timestamp)
        if idx < len(sorted_timestamps):
            after = abs(sorted_timestamps[idx] - target_timestamp)
        else:
            after = before
            assert idx == len(sorted_timestamps), 'wrong imu data'
        return idx - 1 if before <= after else idx


def loadTimestamps(ts_dir):
    # ''' load timestamps '''

    # with open(os.path.join(ts_dir, 'timestamps.txt')) as f:
    #     data=f.read().splitlines()
    # ts = [l.split(' ')[0] for l in data] 

    """source pykitti raw: Load timestamps from file."""
    timestamp_file = os.path.join( ts_dir, 'timestamps.txt')

    # Read and parse the timestamps
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f.readlines():
            # NB: datetime only supports microseconds, but KITTI timestamps
            # give nanoseconds, so need to truncate last 4 characters to
            # get rid of \n (counts as 1) and extra 3 digits
            t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(t)

    # # TODO: Subselect the chosen range of frames, if any
    # if frames is not None:
    #     timestamps = [timestamps[i] for i in frames]

    return timestamps


def loadOxtsData(oxts_dir):
    ''' source kitti360 devkit: reads GPS/IMU data from files to memory. requires base directory
    (=sequence directory as parameter). if frames is not specified, loads all frames. '''

    ts = []
    ts = loadTimestamps(oxts_dir)
    oxts  = []
    for i in range(len(ts)):
      if len(ts[i]):
        try:
          oxts.append(np.loadtxt(os.path.join(oxts_dir, 'data', '%010d.txt'%i)))
        except:
          oxts.append([])
      else:
        oxts.append([])
    return oxts,ts

def read_point_cloud(filename: str, color_channel: int = 0, bin_channel_count: int = 4) -> np.ndarray:

    # read point cloud from either (*.ply, *.pcd, *.las) or (kitti *.bin) format
    if ".bin" in filename:
        # we also read the intensity channel here
        data_loaded = np.fromfile(filename, dtype=np.float32)

        # print(data_loaded)
        # for NCLT, it's a bit different from KITTI format, check: http://robots.engin.umich.edu/nclt/python/read_vel_sync.py
        # for KITTI, bin_channel_count = 4
        # for Boreas, bin_channel_count = 6 # (x,y,z,i,r,ts)
        points = data_loaded.reshape((-1, bin_channel_count))
        # print(points)
        ts = None # KITTI
        if bin_channel_count == 6:
            ts = points[:, -1]
        
    elif ".ply" in filename:
        pc_load = o3d.t.io.read_point_cloud(filename)
        pc_load = {k: v.numpy() for k,v in pc_load.point.items() }
        
        keys = list(pc_load.keys())
        # print("available attributes:", keys)

        points = pc_load['positions']

        if 't' in keys:
            ts = pc_load['t'] * 1e-8
        elif 'timestamp' in keys:
            ts = pc_load['timestamp']
        else:
            ts = None

        if 'colors' in keys and color_channel == 3:           
            colors = pc_load['colors'] # if they are available
            points = np.hstack((points, colors))
        elif 'intensity' in keys and color_channel == 1:           
            intensity = pc_load['intensity'] # if they are available
            # print(intensity)
            points = np.hstack((points, intensity))
    elif ".pcd" in filename: # currently cannot be readed by o3d.t.io
        pc_load = o3d.io.read_point_cloud(filename)
        points = np.asarray(pc_load.points, dtype=np.float64)
        ts = None
    elif ".las" in filename: # use laspy
        import laspy
        with laspy.open(filename) as fh:     
            las = fh.read()
            x = (las.points.X * las.header.scale[0] + las.header.offset[0])
            y = (las.points.Y * las.header.scale[1] + las.header.offset[1])
            z = (las.points.Z * las.header.scale[2] + las.header.offset[2])
            points = np.array([x, y, z], dtype=np.float64).T
            if color_channel == 1:
                intensity = np.array(las.points.intensity).reshape(-1, 1)
                # print(intensity)
                points = np.hstack((points, intensity))
            ts = None # TODO, also read the point-wise timestamp for las point cloud
    else:
        sys.exit("The format of the imported point cloud is wrong (support only *pcd, *ply, *las and *bin)")

    # print("Loaded ", np.shape(points)[0], " points")

    return points, ts # as np

# now we only support semantic kitti format dataset
def read_semantic_point_label(bin_filename: str, label_filename: str, color_on: bool = False):

    # read point cloud (kitti *.bin format)
    if ".bin" in bin_filename:
        # we also read the intensity channel here
        points = np.fromfile(bin_filename, dtype=np.float32).reshape(-1, 4)
    else:
        sys.exit("The format of the imported point cloud is wrong (support only *bin)")

    # read point cloud labels (*.label format)
    if ".label" in label_filename:
        labels = np.fromfile(label_filename, dtype=np.uint32).reshape(-1)
    else:
        sys.exit("The format of the imported point labels is wrong (support only *label)")

    labels = labels & 0xFFFF # only take the semantic part 

    # get the reduced label [0-20] 
    labels_reduced = np.vectorize(sem_map_function)(labels).astype(np.int32)  # fast version

    # original label [0-255]
    labels = np.array(labels, dtype=np.int32)

    return points, labels, labels_reduced # as np

def read_kitti_format_calib(filename: str):
    """ 
        read calibration file (with the kitti format)
        returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    calib_file = open(filename)

    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))

        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    return calib

def read_kitti_format_poses(filename: str) -> List[np.ndarray]:
    """ 
        read pose file (with the kitti format)
        returns -> list, transformation before calibration transformation
    """
    pose_file = open(filename)

    poses = []

    for line in pose_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(pose)

    pose_file.close()
    return poses   

def read_tum_format_poses_csv(filename: str) -> List[np.ndarray]:
    # now it supports csv file only (TODO)
    from pyquaternion import Quaternion

    poses = []
    with open(filename, mode="r") as f:
        reader = csv.reader(f)
        # get header and change timestamp label name
        header = next(reader)
        header[0] = "ts"
        # Convert string odometry to numpy transfor matrices
        for row in reader:
            odom = {l: row[i] for i, l in enumerate(header)}
            # Translarion and rotation quaternion as numpy arrays
            trans = np.array([float(odom[l]) for l in ["tx", "ty", "tz"]])
            quat_ijkw = np.array([float(odom[l]) for l in ["qx", "qy", "qz", "qw"]])
            quat = Quaternion(quat_ijkw[3], quat_ijkw[0], quat_ijkw[1], quat_ijkw[2]) # quaternion needs to use the w, i, j, k order , you need to switch as bit
            rot = quat.rotation_matrix
            # Build numpy transform matrix
            odom_tf = np.eye(4)
            odom_tf[0:3, 3] = trans
            odom_tf[0:3, 0:3] = rot
            # Add transform to timestamp indexed dictionary
            # odom_tfs[odom["ts"]] = odom_tf
            poses.append(odom_tf)

    return poses

# copyright: Nacho et al. KISS-ICP
def write_kitti_format_poses(filename: str, poses: List[np.ndarray]):
    def _to_kitti_format(poses: np.ndarray) -> np.ndarray:
        return np.array([np.concatenate((pose[0], pose[1], pose[2])) for pose in poses])

    np.savetxt(fname=f"{filename}_kitti.txt", X=_to_kitti_format(poses))

# copyright: Nacho et al. KISS-ICP
def write_tum_format_poses(filename: str, poses: List[np.ndarray], timestamps = None):
    from pyquaternion import Quaternion
    def _to_tum_format(poses, timestamps = None):
        tum_data = []
        with contextlib.suppress(ValueError):
            for idx in range(len(poses)):
                tx, ty, tz = poses[idx][:3, -1].flatten()
                qw, qx, qy, qz = Quaternion(matrix=poses[idx], atol=0.01).elements
                if timestamps is None:
                    tum_data.append([idx, tx, ty, tz, qx, qy, qz, qw]) # index as the ts
                else:
                    tum_data.append([float(timestamps[idx]), tx, ty, tz, qx, qy, qz, qw])
        return np.array(tum_data).astype(np.float64)

    np.savetxt(fname=f"{filename}_tum.txt", X=_to_tum_format(poses, timestamps), fmt="%.4f")

def apply_kitti_format_calib(poses: List[np.ndarray], calib_T_cl) -> List[np.ndarray]:
    """Converts from Velodyne to Camera Frame (# T_camera<-lidar)""" 
    poses_calib = []
    if calib_T_cl is not None:
        for pose in poses:
            poses_calib.append(calib_T_cl @ pose @ inv(calib_T_cl))
    return poses_calib 
    
# torch version
def crop_frame(points: torch.tensor, ts: torch.tensor, 
               min_z_th=-3.0, max_z_th=100.0, min_range=2.75, max_range=100.0):
    dist = torch.norm(points[:,:3], dim=1)
    filtered_idx = (dist > min_range) & (dist < max_range) & (points[:, 2] > min_z_th) & (points[:, 2] < max_z_th)
    points = points[filtered_idx]
    if ts is not None:
        ts = ts[filtered_idx]
    return points, ts

# torch version
def intrinsic_correct(points: torch.tensor, correct_deg=0.):

    # # This function only applies for the KITTI dataset, and should NOT be used by any other dataset,
    # # the original idea and part of the implementation is taking from CT-ICP(Although IMLS-SLAM
    # # Originally introduced the calibration factor)
    if correct_deg == 0.:
        return points

    dist = torch.norm(points[:,:3], dim=1)
    kitti_var_vertical_ang = correct_deg / 180. * math.pi
    v_ang = torch.asin(points[:, 2] / dist)
    v_ang_c = v_ang + kitti_var_vertical_ang
    hor_scale = torch.cos(v_ang_c) / torch.cos(v_ang)
    points[:, 0] *= hor_scale
    points[:, 1] *= hor_scale
    points[:, 2] = dist * torch.sin(v_ang_c)

    return points

# now only work for semantic kitti format dataset # torch version
def filter_sem_kitti(points: torch.tensor, sem_labels_reduced: torch.tensor, sem_labels: torch.tensor,
                     filter_outlier = True, filter_moving = False):
    
    # sem_labels_reduced is the reduced labels for mapping (20 classes for semantic kitti)
    # sem_labels is the original semantic label (0-255 for semantic kitti)
    
    if filter_outlier: # filter the outliers according to semantic labels
        inlier_mask = (sem_labels > 1) # not outlier
    else:
        inlier_mask = (sem_labels >= 0) # all

    if filter_moving:
        static_mask = sem_labels < 100 # only for semantic KITTI dataset
        inlier_mask = inlier_mask & static_mask

    points = points[inlier_mask]
    sem_labels_reduced = sem_labels_reduced[inlier_mask]

    return points, sem_labels_reduced

def write_traj_as_o3d(poses: List[np.ndarray], path):

    o3d_pcd = o3d.geometry.PointCloud()
    poses_np = np.array(poses, dtype=np.float64)
    o3d_pcd.points = o3d.utility.Vector3dVector(poses_np[:,:3,3])

    ts_np = np.linspace(0, 1, len(poses))
    color_map = cm.get_cmap('jet')
    ts_color = color_map(ts_np)[:, :3].astype(np.float64)
    o3d_pcd.colors = o3d.utility.Vector3dVector(ts_color)

    if path is not None:
        o3d.io.write_point_cloud(path, o3d_pcd)

    return o3d_pcd

def plot_frame(ax, R, t, frame_label, origin_color, colors, length=0.01):
    # Plot the origin
    ax.scatter(t[0], t[1], t[2], color=origin_color, label=frame_label)
    
    # Plot the x, y, z axes with different colors
    for i in range(3):
        ax.quiver(t[0], t[1], t[2], R[0, i], R[1, i], R[2, i], color=colors[i], length=length)
    
    # Add text labels for the axes
    ax.text(t[0] + R[0, 0] * length, t[1] + R[1, 0] * length, t[2] + R[2, 0] * length, 'X', color=colors[0])
    ax.text(t[0] + R[0, 1] * length, t[1] + R[1, 1] * length, t[2] + R[2, 1] * length, 'Y', color=colors[1])
    ax.text(t[0] + R[0, 2] * length, t[1] + R[1, 2] * length, t[2] + R[2, 2] * length, 'Z', color=colors[2])

def quaternion_to_rotation_matrix(quaternion):
    x, y, z, w = quaternion
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])
    return R

def create_homogeneous_transform(translation, rotation):
    R = quaternion_to_rotation_matrix(rotation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T