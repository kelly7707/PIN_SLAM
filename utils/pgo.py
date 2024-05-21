#!/usr/bin/env python3
# @file      pgo.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import numpy as np
import gtsam
import matplotlib.pyplot as plt
from rich import print
import datetime as dt
import os

from utils.config import Config
from dataset.slam_dataset import SLAMDataset
    
class PoseGraphManager:
    def __init__(self, config: Config, dataset: SLAMDataset):

        self.config = config

        self.silence = config.silence

        self.fixed_cov = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9])) # fixed

        tran_std = config.pgo_tran_std # m
        rot_std = config.pgo_rot_std # degree # better to be small

        self.const_cov = np.array([np.radians(rot_std), np.radians(rot_std), np.radians(rot_std), tran_std, tran_std, tran_std]) # first rotation, then translation
        self.odom_cov = gtsam.noiseModel.Diagonal.Sigmas(self.const_cov)
        self.loop_cov = gtsam.noiseModel.Diagonal.Sigmas(self.const_cov)
        
        # not used
        # mEst = gtsam.noiseModel.mEstimator.GemanMcClure(1.0)
        # self.robust_loop_cov = gtsam.noiseModel.Robust(mEst, self.loop_cov)
        # self.robust_odom_cov = gtsam.noiseModel.Robust(mEst, self.odom_cov)

        self.graph_factors = gtsam.NonlinearFactorGraph() # edges # with pose and pose covariance
        self.graph_initials = gtsam.Values() # initial guess # as pose

        self.cur_pose = None
        self.curr_node_idx = None
        self.graph_optimized = None

        self.init_poses = []
        self.pgo_poses = []
        self.loop_edges = []

        self.min_loop_idx = config.end_frame+1
        self.last_loop_idx = 0
        self.drift_radius = 0.0 # m
        self.pgo_count = 0

        self.imu_Preintegration_init(dataset)

    def imu_Preintegration_init(self, dataset: SLAMDataset):
        # source gtsam.CombinedImuFactorExample.py
        self.GRAVITY = 9.81
        params = gtsam.PreintegrationCombinedParams.MakeSharedD(self.GRAVITY) # TODO OXTS coordinates are defined as x = forward, y = right, z = down
        
        self.accBias, self.gyroBias, accel_sigma, gyro_sigma = self.imu_calibration(dataset)
        self.imu_bias = gtsam.imuBias.ConstantBias(self.accBias, self.gyroBias)
        # Some arbitrary noise sigmas
        # gyro_sigma = 1e-3
        # accel_sigma = 1e-3
        I_3x3 = np.eye(3)
        params.setGyroscopeCovariance(gyro_sigma**2 * I_3x3)
        params.setAccelerometerCovariance(accel_sigma**2 * I_3x3)
        params.setIntegrationCovariance(1e-7**2 * I_3x3)

        self.pim = gtsam.PreintegratedCombinedMeasurements(params, self.imu_bias)

    # --- IMU
    def preintegration(self, acc, gyro, dts, last_pose, cur_id: int):
        #
        for i,dt in enumerate(dts):
            # self.pim.integrate() # https://github.com/borglab/gtsam/blob/0fee5cb76e7a04b590ff0dc1051950da5b265340/python/gtsam/examples/PreintegrationExample.py#L159C16-L159C70 
            self.pim.integrateMeasurement(acc[i], gyro[i], dt) # https://github.com/borglab/gtsam/blob/4abef9248edc4c49943d8fd8a84c028deb486f4c/python/gtsam/examples/CombinedImuFactorExample.py#L175C12-L177C74 
        # preintegration
        initial_state = gtsam.NavState(
            last_pose,
            self.velocity) # https://github.com/borglab/gtsam/blob/4abef9248edc4c49943d8fd8a84c028deb486f4c/python/gtsam/examples/CombinedImuFactorExample.py#L164C9-L168C46 
        imu_prediction = self.pim.predict(initial_state, self.imu_bias)
        predicted_pose = imu_prediction.pose()
        self.velocity = imu_prediction.velocity()

        self.graph_initials.insert(gtsam.symbol('v', cur_id), self.velocity)
        self.graph_initials.insert(gtsam.symbol('b', cur_id),  ) # https://github.com/borglab/gtsam/blob/4abef9248edc4c49943d8fd8a84c028deb486f4c/python/gtsam/examples/CombinedImuFactorExample.py#L219C17-L221C55 

        
    
    def add_combined_IMU_factor(self,cur_id: int, last_id: int): 
        #
        self.graph_factors.add(gtsam.CombinedImuFactor(gtsam.symbol('x', last_id), gtsam.symbol('v', last_id), gtsam.symbol('x', cur_id),
                                gtsam.symbol('v', cur_id), gtsam.symbol('b', last_id), gtsam.symbol('b', cur_id), self.pim))
    
    def imu_calibration(self, dataset: SLAMDataset, imu_calibration_steps=30, imu_calibration_time = 3.0, visual=False):
        #
        num_samples = 0
        gyro_avg = np.zeros(3)
        accel_avg = np.zeros(3)

        ang_vel_list = []
        lin_acc_list = []
        timestamps = []

        # for idx in range(int(10*imu_calibration_time)):
        #     if dataset.ts_rawimu[idx].timestamp() - dataset.ts_rawimu[0].timestamp() < imu_calibration_time:
        
        for idx in range(imu_calibration_steps):
            num_samples+=1

            raw_imu_filename = os.path.join(self.config.raw_imu_path, 'data', f'{idx:010d}.txt') #w-17:20, a-11:14
            raw_imu = np.loadtxt(raw_imu_filename)
            lin_acc = raw_imu[11:14]
            ang_vel = raw_imu[17:20]

            gyro_avg += ang_vel
            accel_avg += lin_acc

            ang_vel_list.append(ang_vel)
            lin_acc_list.append(lin_acc)
            timestamps.append(dataset.ts_rawimu[idx].timestamp())

        gyro_avg /= num_samples
        accel_avg /= num_samples
        grav_vec = np.array([0, 0, self.GRAVITY])
        
        accBias = accel_avg - grav_vec
        gyroBias = gyro_avg

        ang_vel_array = np.array(ang_vel_list)
        lin_acc_array = np.array(lin_acc_list)
        timestamps = np.array(timestamps)
        # Calculate max-min range for sigma initialization
        gyro_sigma = (ang_vel_array.max(axis=0) - ang_vel_array.min(axis=0)) / 2
        accel_sigma = (lin_acc_array.max(axis=0) - lin_acc_array.min(axis=0)) / 2
        
        # V- Plotting the IMU data
        if visual:            
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            # Gyroscope data
            axs[0].plot(timestamps, ang_vel_array[:, 0], label='Gyro X', color='r')
            axs[0].plot(timestamps, ang_vel_array[:, 1], label='Gyro Y', color='g')
            axs[0].plot(timestamps, ang_vel_array[:, 2], label='Gyro Z', color='b')
            axs[0].axhline(gyroBias[0], color='r', linestyle='--', label='Bias X')
            axs[0].axhline(gyroBias[1], color='g', linestyle='--', label='Bias Y')
            axs[0].axhline(gyroBias[2], color='b', linestyle='--', label='Bias Z')
            axs[0].set_title('Gyroscope Data')
            axs[0].set_xlabel('Time [s]')
            axs[0].set_ylabel('Angular Velocity [rad/s]')
            axs[0].legend()
            axs[0].grid(True)
            # Display gyro sigmas on the plot
            axs[0].text(0.05, 0.95, f'Gyro Sigma X: {gyro_sigma[0]:.6f}\nGyro Sigma Y: {gyro_sigma[1]:.6f}\nGyro Sigma Z: {gyro_sigma[2]:.6f}',
                    transform=axs[0].transAxes, fontsize=10, verticalalignment='top')
            # Accelerometer data
            axs[1].plot(timestamps, lin_acc_array[:, 0], label='Accel X', color='r')
            axs[1].plot(timestamps, lin_acc_array[:, 1], label='Accel Y', color='g')
            axs[1].plot(timestamps, lin_acc_array[:, 2], label='Accel Z', color='b')
            axs[1].axhline(accBias[0], color='r', linestyle='--', label='Bias X + Gravity')
            axs[1].axhline(accBias[1], color='g', linestyle='--', label='Bias Y')
            axs[1].axhline(accBias[2] + self.GRAVITY, color='b', linestyle='--', label='Bias Z')
            axs[1].set_title('Accelerometer Data')
            axs[1].set_xlabel('Time [s]')
            axs[1].set_ylabel('Linear Acceleration [m/s^2]')
            axs[1].legend()
            axs[1].grid(True)
            # Display accel sigmas on the plot
            axs[1].text(0.05, 0.95, f'Accel Sigma X: {accel_sigma[0]:.6f}\nAccel Sigma Y: {accel_sigma[1]:.6f}\nAccel Sigma Z: {accel_sigma[2]:.6f}',
                    transform=axs[1].transAxes, fontsize=10, verticalalignment='top')

            plt.tight_layout()
            plt.show()

        return accBias, gyroBias, accel_sigma, gyro_sigma
    
    
    def add_frame_node(self, frame_id, init_pose):
        """create frame pose node and set pose initial guess  
        Args:
            frame_id: int
            init_pose (np.array): 4x4, as T_world<-cur
        """
        self.curr_node_idx = frame_id # make start with 0
        if not self.graph_initials.exists(gtsam.symbol('x', frame_id)): # create if not yet exists
            self.graph_initials.insert(gtsam.symbol('x', frame_id), gtsam.Pose3(init_pose))
            # v b 
        
    def add_pose_prior(self, frame_id: int, prior_pose: np.ndarray, fixed: bool = False):
        """add pose prior unary factor  
        Args:
            frame_id: int
            prior_pose (np.array): 4x4, as T_world<-cur
            dist_ratio: float , use to determine the covariance, the std is porpotional to this dist_ratio
            fixed: bool, if True, this frame is fixed with very low covariance
        """

        if fixed:
            cov_model = self.fixed_cov
        else:
            tran_sigma = self.drift_radius+1e-4 # avoid divide by 0
            rot_sigma = self.drift_radius * np.radians(10.0)
            cov_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([rot_sigma, rot_sigma, rot_sigma, tran_sigma, tran_sigma, tran_sigma]))
      
        self.graph_factors.add(gtsam.PriorFactorPose3(
                                            gtsam.symbol('x', frame_id), 
                                            gtsam.Pose3(prior_pose), 
                                            cov_model))

    def add_odometry_factor(self, cur_id: int, last_id: int, odom_transform: np.ndarray, cov = None):
        """! add a odometry factor between two adjacent pose nodes
        Args:
            cur_id: int
            last_id: int
            odom_transform (np.array): 4x4 , as T_prev<-cur
            cov (np.array): 6x6 covariance matrix, if None, set to the default value
        """
        
        if cov is None:
            cov_model = self.odom_cov
        else:
            cov_model = gtsam.noiseModel.Gaussian.Covariance(cov)

        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                                gtsam.symbol('x', last_id), #t-1
                                                gtsam.symbol('x', cur_id),  #t
                                                gtsam.Pose3(odom_transform),  # T_prev<-cur
                                                cov_model))  # NOTE: add robust kernel
        # TODO: V,B
    
    
    def add_loop_factor(self, cur_id: int, loop_id: int, loop_transform: np.ndarray, cov = None):
        """add a loop closure factor between two pose nodes
        Args:
            cur_id: int
            loop_id: int
            loop_transform (np.array): 4x4 , as T_loop<-cur
            cov (np.array): 6x6 covariance matrix, if None, set to the default value
        """

        if cov is None:
            cov_model = self.loop_cov
        else:
            cov_model = gtsam.noiseModel.Gaussian.Covariance(cov)

        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                                gtsam.symbol('x', loop_id), #l
                                                gtsam.symbol('x', cur_id),  #t 
                                                gtsam.Pose3(loop_transform),  # T_loop<-cur
                                                cov_model))  # NOTE: add robust kernel

    def optimize_pose_graph(self):
        
        if self.config.pgo_with_lm: # default
            opt_param = gtsam.LevenbergMarquardtParams()
            opt_param.setMaxIterations(self.config.pgo_max_iter)
            opt = gtsam.LevenbergMarquardtOptimizer(self.graph_factors, self.graph_initials, opt_param)
        else: # pgo with dogleg
            opt_param = gtsam.DoglegParams()
            opt_param.setMaxIterations(self.config.pgo_max_iter)
            opt = gtsam.DoglegOptimizer(self.graph_factors, self.graph_initials, opt_param)    
        
        self.graph_optimized = opt.optimizeSafely()

        # Calculate marginal covariances for all variables
        # marginals = gtsam.Marginals(self.graph_factors, self.graph_optimized)
        # try to even visualize the covariance
        # cov = get_node_cov(marginals, 50)
        # print(cov)

        error_before = self.graph_factors.error(self.graph_initials)
        error_after = self.graph_factors.error(self.graph_optimized)
        if not self.silence:
            print("[bold red]PGO done[/bold red]")
            print("error %f --> %f:" % (error_before, error_after))

        self.graph_initials = self.graph_optimized # update the initial guess

        # update the pose of each frame after pgo
        frame_count = self.curr_node_idx+1
        self.pgo_poses = [None] * frame_count # start from 0
        for idx in range(frame_count):
            self.pgo_poses[idx] = get_node_pose(self.graph_optimized, idx)

        self.cur_pose = self.pgo_poses[self.curr_node_idx] 

        self.pgo_count += 1


        self.pim.resetIntegration()

    def write_g2o(self, out_file):
        gtsam.writeG2o(self.graph_factors, self.graph_initials, out_file)

    def get_pose_diff(self):

        assert len(self.pgo_poses) == len(self.init_poses), "Lists of poses must have the same size."
        pose_diff = np.array([(pgo_pose @ np.linalg.inv(init_pose)) for pgo_pose, init_pose in zip(self.pgo_poses, self.init_poses)])
        return pose_diff
    
    def estimate_drift(self, travel_dist_list, used_frame_id, drfit_ratio = 0.01, correct_ratio = 0.01):
        # estimate the current drift # better to calculate according to residual
        self.drift_radius = (travel_dist_list[used_frame_id] - travel_dist_list[self.last_loop_idx])*drfit_ratio
        if self.min_loop_idx < self.last_loop_idx: # the loop has been corrected previously
            self.drift_radius += (travel_dist_list[self.min_loop_idx] + travel_dist_list[used_frame_id]*correct_ratio)*drfit_ratio
        # print("Estimated drift (m):", self.drift_radius)
    
    def plot_loops(self, loop_plot_path, vis_now = False):
    
        pose_count = len(self.pgo_poses)
        z_ratio = 0.002
        ts = np.arange(0, pose_count, 1) * z_ratio

        traj_est = np.vstack([pose[:3, 3] for pose in self.pgo_poses])
        # print(poses_est)

        # Create a 3D plot
        fig = plt.figure() # facecolor='white'
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory
        ax.plot(traj_est[:,0], traj_est[:,1], ts, 'k')

        for loop in self.loop_edges:
            node_0 = loop[0]
            node_1 = loop[1]
            ax.plot([traj_est[node_0, 0], traj_est[node_1, 0]], [traj_est[node_0, 1], traj_est[node_1, 1]], [ts[node_0], ts[node_1]], color='green')

        # TODO: Set labels and title

        ax.grid(False)
        ax.set_axis_off()
        # turn of the gray background
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_box_aspect([1, 1, 1])

        plt.tight_layout()

        if loop_plot_path is not None:
            plt.savefig(loop_plot_path, dpi=600)

        # Show the plot
        if vis_now:
            plt.show()


    
    

def get_node_pose(graph, idx):

    pose = graph.atPose3(gtsam.symbol('x', idx))
    # print(pose)
    pose_se3 = np.eye(4)
    pose_se3[:3, 3] = np.array([pose.x(), pose.y(), pose.z()])
    pose_se3[:3, :3] = pose.rotation().matrix()

    return pose_se3

def get_node_cov(marginals, idx):

    cov = marginals.marginalCovariance(gtsam.symbol('x', idx))
    # print(cov)
    # pose_se3 = np.eye(4)
    # pose_se3[:3, 3] = np.array([pose.x(), pose.y(), pose.z()])
    # pose_se3[:3, :3] = pose.rotation().matrix()

    return cov


