#!/usr/bin/env python3
# @file      data_sampler.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import torch
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

from utils.config import Config
from utils.tools import get_time

class DataSampler():

    def __init__(self, config: Config):

        self.config = config
        self.dev = config.device


    # input and output are all torch tensors
    def sample(self, points_torch, 
               normal_torch,
               sem_label_torch,
               color_torch, normal_guided_sampling = False):
        # points_torch is in the sensor's local coordinate system, not yet transformed to the global system

        # self.visualize_point_cloud(points_torch,normal_torch)

        T0 = get_time()

        dev = self.dev

        surface_sample_range = self.config.surface_sample_range_m #0.3
        surface_sample_n = self.config.surface_sample_n # 3
        freespace_behind_sample_n = self.config.free_behind_n # 1
        freespace_front_sample_n = self.config.free_front_n # 2

        all_sample_n = surface_sample_n+freespace_behind_sample_n+freespace_front_sample_n+1 # 1 as the exact measurement #default ~=7
        free_front_min_ratio = self.config.free_sample_begin_ratio # 0.3
        free_sample_end_dist = self.config.free_sample_end_dist_m # 1
        # clearance_dist_scaled = self.config.clearance_dist_m * self.config.scale
        
        sigma_base = self.config.sigma_sigmoid_m #0.08

        # get sample points
        point_num = points_torch.shape[0]
        distances = torch.linalg.norm(points_torch, dim=1, keepdim=True) # ray distances (scaled) # in local frame (distance from sensor to each point)
        
        # Part 0. the exact measured point
        measured_sample_displacement = torch.zeros_like(distances)
        measured_sample_dist_ratio = torch.ones_like(distances)

        # Part 1. close-to-surface uniform sampling 
        # uniform sample in the close-to-surface range (+- range)
        surface_sample_displacement = torch.randn(point_num*surface_sample_n, 1, device=dev)*surface_sample_range  # Generates a tensor of random values drawn from a normal distribution (mean=0, std=1) * scales the displacements to the desired range --> creates random displacements around the surface of the objects.
        
        repeated_dist = distances.repeat(surface_sample_n,1)
        surface_sample_dist_ratio = surface_sample_displacement/repeated_dist + 1.0 # (dist_ratio) 1.0 means on the surface
        if sem_label_torch is not None:
            surface_sem_label_tensor = sem_label_torch.repeat(1, surface_sample_n).transpose(0,1)
        if color_torch is not None:
            color_channel = color_torch.shape[1]
            surface_color_tensor = color_torch.repeat(surface_sample_n,1)
        
        # deprecated
        # # near surface uniform sampling (for clearance) [from the close surface lower bound closer to the sensor for a clearance distance]
        # clearance_sample_displacement = -torch.rand(point_num*clearance_sample_n, 1, device=dev)*clearance_dist_scaled - surface_sample_range_scaled

        # repeated_dist = distances.repeat(clearance_sample_n,1)
        # clearance_sample_dist_ratio = clearance_sample_displacement/repeated_dist + 1.0 # 1.0 means on the surface
        # if sem_label_torch is not None:
        #     clearance_sem_label_tensor = torch.zeros_like(repeated_dist)

        # Part 2. free space (front) uniform sampling
        # if you want to reconstruct the thin objects (like poles, tree branches) well, you need more freespace samples to have 
        # a space carving effect

        sigma_ratio = 2.0

        repeated_dist = distances.repeat(freespace_front_sample_n,1)
        free_max_ratio = 1.0 - sigma_ratio * surface_sample_range / repeated_dist
        free_diff_ratio = free_max_ratio - free_front_min_ratio

        free_sample_front_dist_ratio = torch.rand(point_num*freespace_front_sample_n, 1, device=dev)*free_diff_ratio + free_front_min_ratio
        
        free_sample_front_displacement = (free_sample_front_dist_ratio - 1.0) * repeated_dist
        if sem_label_torch is not None:
            free_sem_label_front = torch.zeros_like(repeated_dist)
        if color_torch is not None:
            free_color_front = torch.zeros(point_num*freespace_front_sample_n, color_channel, device=dev)

        # Part 3. free space (behind) uniform sampling
        repeated_dist = distances.repeat(freespace_behind_sample_n,1)
        free_max_ratio = free_sample_end_dist / repeated_dist + 1.0
        free_behind_min_ratio = 1.0 + sigma_ratio * surface_sample_range / repeated_dist
        free_diff_ratio = free_max_ratio - free_behind_min_ratio

        free_sample_behind_dist_ratio = torch.rand(point_num*freespace_behind_sample_n, 1, device=dev)*free_diff_ratio + free_behind_min_ratio
        
        free_sample_behind_displacement = (free_sample_behind_dist_ratio - 1.0) * repeated_dist
        if sem_label_torch is not None:
            free_sem_label_behind = torch.zeros_like(repeated_dist)
        if color_torch is not None:
            free_color_behind = torch.zeros(point_num*freespace_behind_sample_n, color_channel, device=dev)
        
        T1 = get_time()

        # all together
        all_sample_displacement = torch.cat((measured_sample_displacement, surface_sample_displacement, free_sample_front_displacement, free_sample_behind_displacement),0) # around 0
        all_sample_dist_ratio = torch.cat((measured_sample_dist_ratio, surface_sample_dist_ratio, free_sample_front_dist_ratio, free_sample_behind_dist_ratio),0) # around 1
        
        repeated_points = points_torch.repeat(all_sample_n,1)
        repeated_dist = distances.repeat(all_sample_n,1)
        # -- sample points
        # if normal_guided_sampling: 
        #     normal_direction = normal_torch.repeat(all_sample_n,1) # normals are oriented towards sensors.
        #     #note that normals are oriented towards origin (inwards)
        #     all_sample_points = repeated_points + all_sample_displacement * (-normal_direction)
        # else: # pin-slam
        #     all_sample_points = repeated_points*all_sample_dist_ratio
        #     # all_sample_points = repeated_points*all_sample_dist_ratio + sensor_origin_torch

        # -- sample points (only add normal guided sampling for the close-to-surface samples)
        if normal_guided_sampling: 
            measured_sample_points = points_torch

            surface_normal_direction = normal_torch.repeat(surface_sample_n,1)
            surface_points = points_torch.repeat(surface_sample_n,1)
            surface_sample_points = surface_points+ surface_sample_displacement * (-surface_normal_direction)
            
            free_front_points = points_torch.repeat(freespace_front_sample_n,1) 
            free_front_sample_points = free_front_points * free_sample_front_dist_ratio
            
            free_behind_points = points_torch.repeat(freespace_behind_sample_n,1)
            free_behind_sample_points = free_behind_points * free_sample_behind_dist_ratio

            all_sample_points = torch.cat((measured_sample_points, surface_sample_points, free_front_sample_points, free_behind_sample_points),0)
        else:
            all_sample_points = repeated_points*all_sample_dist_ratio

        # depth tensor of all the samples
        depths_tensor = repeated_dist * all_sample_dist_ratio

        # linear error model: sigma(d) = sigma_base + d * sigma_scale_constant
        # ray_sigma = sigma_base + distances * sigma_scale_constant  
        # different sigma value for different ray with different distance (deprecated)
        # sigma_tensor = ray_sigma.repeat(all_sample_n,1).squeeze(1)

        # get the weight vector as the inverse of sigma
        weight_tensor = torch.ones_like(depths_tensor)

        surface_sample_count = point_num*(surface_sample_n+1)
        if self.config.dist_weight_on: # far away surface samples would have lower weight
            weight_tensor[:surface_sample_count] = 1 + self.config.dist_weight_scale*0.5 - (repeated_dist[:surface_sample_count] / self.config.max_range) * self.config.dist_weight_scale # [0.6, 1.4]
        # TODO: also add lower weight for surface samples with large incidence angle

        # behind surface weight drop-off because we have less uncertainty behind the surface
        if self.config.behind_dropoff_on:
            dropoff_min = 0.2 * free_sample_end_dist
            dropoff_max = free_sample_end_dist
            dropoff_diff = dropoff_max - dropoff_min
            # behind_displacement = (repeated_dist*(all_sample_dist_ratio-1.0)/sigma_base)
            behind_displacement = all_sample_displacement
            dropoff_weight = (dropoff_max - behind_displacement) / dropoff_diff
            dropoff_weight = torch.clamp(dropoff_weight, min = 0.0, max = 1.0)
            dropoff_weight = dropoff_weight * 0.8 + 0.2
            weight_tensor = weight_tensor * dropoff_weight
        
        # give a flag indicating the type of the sample [negative: freespace, positive: surface]
        weight_tensor[surface_sample_count:] *= -1.0 
        
        # ray-wise depth
        distances = distances.squeeze(1)

        # assign sdf labels to the samples
        # projective distance as the label: behind +, in-front - 
        sdf_label_tensor = all_sample_displacement.squeeze(1)  # scaled [-1, 1] # as distance (before sigmoid)
        # assert torch.all(sdf_label_tensor <= 1.0) and torch.all(sdf_label_tensor >= -1.0) # not necessary

        # assign the normal label to the samples
        normal_label_tensor = None
        if normal_torch is not None:
            normal_label_tensor = normal_torch.repeat(all_sample_n,1)
        
        # assign the semantic label to the samples (including free space as the 0 label)
        sem_label_tensor = None
        if sem_label_torch is not None:
            sem_label_tensor = torch.cat((sem_label_torch.unsqueeze(-1), surface_sem_label_tensor, free_sem_label_front, free_sem_label_behind),0).int()

        color_tensor = None
        if color_torch is not None:
            color_tensor = torch.cat((color_torch, surface_color_tensor, free_color_front, free_color_behind),0)

        T2 = get_time()
        # Convert from the all ray surface + all ray free order to the 
        # ray-wise (surface + free) order
        all_sample_points = all_sample_points.reshape(all_sample_n, -1, 3).transpose(0, 1).reshape(-1, 3)
        sdf_label_tensor = sdf_label_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1) 
        sdf_label_tensor *= (-1) # convert to the same sign as 
        # assert torch.all(sdf_label_tensor <= 1.0) and torch.all(sdf_label_tensor >= -1.0) # not necessary

        
        weight_tensor = weight_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        # depths_tensor = depths_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)

        if normal_torch is not None:
            normal_label_tensor = normal_label_tensor.reshape(all_sample_n, -1, 3).transpose(0, 1).reshape(-1, 3)
        if sem_label_torch is not None:
            sem_label_tensor = sem_label_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        if color_torch is not None:
            color_tensor = color_tensor.reshape(all_sample_n, -1, color_channel).transpose(0, 1).reshape(-1, color_channel)

        # ray distance (distances) is not repeated

        T3 = get_time()

        # print("time for sampling I:", T1-T0)
        # print("time for sampling II:", T2-T1)
        # print("time for sampling III:", T3-T2)
        # all super fast, all together in 0.5 ms

        # self.visualize_sdf_with_colors(all_sample_points, sdf_label_tensor)
        # self.visualize_sdf_with_threshold_and_color(all_sample_points, sdf_label_tensor, threshold1=0.6, threshold2=0.6)
        # self.visualize_point_cloud(all_sample_points, normal_label_tensor)
        
        return all_sample_points, sdf_label_tensor, normal_label_tensor, sem_label_tensor, color_tensor, weight_tensor

    
    def visualize_point_cloud(self, points_torch, normal_torch,scale=1):
        # Convert tensors to numpy arrays
        points_np = points_torch.cpu().numpy()
        normals_np = normal_torch.cpu().numpy()

        # Create an Open3D point cloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_np)
        point_cloud.normals = o3d.utility.Vector3dVector(normals_np)

        # Create line sets to represent normals as arrows/lines
        lines = []
        line_colors = []
        points_with_arrows = []

        for i in range(points_np.shape[0]):
            start_point = points_np[i]
            end_point = start_point + normals_np[i] * scale  # scale to control arrow size
            
            points_with_arrows.append(start_point)
            points_with_arrows.append(end_point)
            
            lines.append([i*2, i*2 + 1])
            line_colors.append([0, 1, 0])  # Green color for normals

        # Create line set object for the normals
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_with_arrows),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(line_colors)

        # Visualize point cloud with normal vectors
        o3d.visualization.draw_geometries([point_cloud, line_set])

    
    def visualize_sdf_with_colors(self, points_torch, sdf_torch):

        # Convert tensors to numpy arrays
        points_np = points_torch.cpu().numpy()
        sdf_np = sdf_torch.cpu().numpy()

        # Normalize the SDF values to be between 0 and 1 for color mapping
        min_sdf, max_sdf = np.min(sdf_np), np.max(sdf_np)
        sdf_normalized = (sdf_np - min_sdf) / (max_sdf - min_sdf)

        # Generate colors based on the normalized SDF values
        colormap = plt.get_cmap('coolwarm')  # 'coolwarm' goes from blue to red
        colors_np = colormap(sdf_normalized)[:, :3]  # Extract only the RGB values

        # Create an Open3D point cloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_np)
        point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

        # Visualize the point cloud with SDF colors
        o3d.visualization.draw_geometries([point_cloud])


    def visualize_sdf_with_threshold_and_color(self, points_torch, sdf_torch, threshold1=0.3, threshold2=0.6):
        # Convert tensors to numpy arrays
        points_np = points_torch.cpu().numpy()
        sdf_np = sdf_torch.cpu().numpy()

        # Filter points that are within the specified SDF thresholds
        mask = (sdf_np <= threshold1) 
        points_in_layer = points_np[mask]
        sdf_in_layer = sdf_np[mask]

        # Ensure there are points in the layer
        if points_in_layer.shape[0] == 0:
            print("No points found within the specified thresholds.")
            return

        # Normalize the SDF values in the layer to be between 0 and 1
        min_sdf, max_sdf = np.min(sdf_in_layer), np.max(sdf_in_layer)
        sdf_normalized = (sdf_in_layer - min_sdf) / (max_sdf - min_sdf)

        # Generate colors based on the normalized SDF values
        colormap = plt.get_cmap('coolwarm')  # 'coolwarm' goes from blue to red
        colors_np = colormap(sdf_normalized)[:, :3]  # Extract RGB values only

        # Create an Open3D point cloud object for the points in the layer
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_in_layer)
        point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

        # Visualize the point cloud for the specified SDF layer
        o3d.visualization.draw_geometries([point_cloud])


    def sample_source_pc(self, points):

        dev = self.dev

        sample_count_per_point = 0 
        sampel_max_range = 0.2

        if sample_count_per_point == 0: # use only the original points
            return points, torch.zeros(points.shape[0], device=dev)
        
        unground_points = points[points[:,2]> -1.5]

        point_num = unground_points.shape[0]

        repeated_points = unground_points.repeat(sample_count_per_point,1)

        surface_sample_displacement = (torch.rand(point_num*sample_count_per_point, 1, device=dev)-0.5)*2*sampel_max_range 
        
        distances = torch.linalg.norm(unground_points, dim=1, keepdim=True) # ray distances 

        repeated_dist = distances.repeat(sample_count_per_point,1)
        sample_dist_ratio = surface_sample_displacement/repeated_dist + 1.0 # 1.0 means on the surface

        sample_points = repeated_points*sample_dist_ratio
        sample_labels = -surface_sample_displacement.squeeze(-1)

        sample_points = torch.cat((points, sample_points), 0)
        sample_labels = torch.cat((torch.zeros(points.shape[0], device=dev), sample_labels), 0)

        return sample_points, sample_labels
