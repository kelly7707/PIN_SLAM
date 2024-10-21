"""Description: This script is used to extract multiple (5s or 50 msgs) sequences from each .bag file, mixing them, 
and using one or two (n) sequences from a single bag per training iteration"""

import rosbag
import random
import numpy as np
import torch
from sensor_msgs import point_cloud2
from torch.utils.data import DataLoader, Dataset, Sampler
import os
import csv

from dataset.slam_dataset import quaternion_to_rotation_matrix

# Read TUM format poses --> torch (T = torch.eye(4, dtype=torch.float64, device=self.device))
def load_tum_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            timestamp = float(row[0])
            translation = np.array([float(row[1]), float(row[2]), float(row[3])])
            quaternion = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
            poses.append((timestamp, translation, quaternion))
    return poses

def create_homogeneous_transform_torch(translation, rotation):
    R = quaternion_to_rotation_matrix(rotation)  # Convert quaternion to rotation matrix
    T = torch.eye(4, dtype=torch.float64)  # Create a 4x4 identity matrix in torch
    T[:3, :3] = torch.tensor(R, dtype=torch.float64)  # Set the rotation matrix
    T[:3, 3] = torch.tensor(translation, dtype=torch.float64)  # Set the translation vector
    return T


def load_multiple_sequences(bag_file_path, gt_poses_file, point_cloud_topic, num_sequences=5, num_msgs=50, cache_dir='data/Pretrain_Data'):
    """Extract multiple random, non-overlapping sequences by selecting 50 consecutive point cloud messages."""
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if cached sequences exist for this bag file
    cache_file = os.path.join(cache_dir, f"{os.path.basename(bag_file_path)}_{num_sequences}_sequences.pt")
    if os.path.exists(cache_file):
        print(f"Loading cached sequences from {cache_file}")
        return torch.load(cache_file)  # Load cached sequences
    

    # Step 1: Preload all message timestamps & GT poses
    gt_poses = load_tum_poses(gt_poses_file)

    timestamps = []
    with rosbag.Bag(bag_file_path, 'r') as bag:
        for _, _, t in bag.read_messages(topics=[point_cloud_topic]):
            timestamps.append(t)

    # Ensure we have enough messages for multiple sequences
    total_messages = len(timestamps)
    if total_messages < num_msgs * num_sequences:
        raise ValueError(f"Not enough data in {bag_file_path} for {num_sequences} sequences of {num_msgs} messages each")

    # Step 2: Randomly select starting indices for non-overlapping sequences
    selected_sequences = []
    selected_poses = [] 
    used_indices = []
    random.seed(42)  # For reproducibility
    for _ in range(num_sequences):
        while True:
            start_idx = random.randint(0, total_messages - num_msgs)

            # Ensure no overlap with previously selected sequences
            overlaps = any(start_idx < end and start_idx + num_msgs > start for start, end in used_indices)

            if not overlaps:
                used_indices.append((start_idx, start_idx + num_msgs))
                break

        # Step 3: Extract point cloud data from the selected indices using time intervals
        point_coords = []
        poses_current_interval = []
        with rosbag.Bag(bag_file_path, 'r') as bag:
            start_time = timestamps[start_idx]
            end_time = timestamps[start_idx + num_msgs - 1]

            # Read messages within the selected time interval
            for topic, msg, t in bag.read_messages(topics=[point_cloud_topic], start_time=start_time, end_time=end_time):
                points = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
                point_coords.append(torch.tensor(points, dtype=torch.float32))

                # Find the closest GT pose for the current timestamp
                # TODO: more efficient (no need to check all poses, and check for the entire sequence)
                # TODO: visualize the pc and gt poses
                closest_pose = min(gt_poses, key=lambda pose: abs(pose[0] - t.to_sec()))
                T = create_homogeneous_transform_torch(closest_pose[1], closest_pose[2])
                poses_current_interval.append(T)

        # Concatenate the point clouds for the current sequence into a tensor
        if point_coords and poses_current_interval:
            selected_sequences.append(torch.stack(point_coords))  # Stack the point cloud tensors for the entire sequence
            selected_poses.append(torch.stack(poses_current_interval))  # Stack poses for the entire sequence

        # Final check to ensure lengths are equal
        if len(selected_sequences) != len(selected_poses):
            raise ValueError("Mismatch in lengths of selected_sequences and selected_poses.")


    # Step 4: Cache the generated sequences
    print(f"Caching sequences to {cache_file}")
    torch.save((selected_sequences, selected_poses), cache_file)
    
    return selected_sequences, selected_poses  # List of point cloud tensors (one per sequence)


class PointCloudDataset(Dataset):
    def __init__(self, sequence_paths, gt_poses_files, point_cloud_topics, num_sequences_per_bag=10, cache_dir='data/Pretrain_Data'):
        self.sequences = []  # Store all sequences, but track which bag they came from
        self.sequence_gt_poses = []  
        self.sequence_labels = []  # Keep track of which .bag each sequence came from
        self.unique_bag_labels = []
        
        # Load multiple sequences from each bag and store them
        for sequence_path, gt_poses_file, point_cloud_topic in zip(sequence_paths, gt_poses_files, point_cloud_topics):
            print('--------loading sequences from bag')
            sequences_from_bag, gt_poses_from_bag = load_multiple_sequences(sequence_path, gt_poses_file, point_cloud_topic, num_sequences=num_sequences_per_bag)
            self.sequences.extend(sequences_from_bag)  # Store sequences from the bag
            self.sequence_gt_poses.extend(gt_poses_from_bag)  
            self.sequence_labels.extend([sequence_path] * num_sequences_per_bag)  # Track source of sequences
            self.unique_bag_labels.append(sequence_path)
        print('--------sequences loaded for all bags')
        # Get unique sequence labels and save them
        self.unique_bag_labels = np.array(self.unique_bag_labels)
        # unique_bag_labels = np.unique(self.unique_bag_labels) #  no need
        
        self.save_unique_sequences(self.unique_bag_labels, f'{cache_dir}/unique_labels.pt')

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Return a sequence and the label (the bag it came from)
        return self.sequences[idx], self.sequence_gt_poses[idx], self.sequence_labels[idx]

    def save_unique_sequences(self, unique_bag_labels, label_file):
        if os.path.exists(label_file):
            os.remove(label_file)  # Delete the existing file
            print(f"Existing file {label_file} deleted.")
        
        print(f"Saving unique sequence labels to {label_file}")
        with open(label_file, 'w') as file:
            for label in unique_bag_labels:
                file.write(f"{label}\n")
        print(f"Unique sequences saved to {label_file}")


class SingleBagBatchSampler(Sampler):
    def __init__(self, dataset, sequences_per_batch=1):
        self.dataset = dataset
        self.sequences_per_batch = sequences_per_batch

    def __iter__(self):
        # Step 1: Convert for efficient comparison
        sequence_labels_np = np.array(self.dataset.sequence_labels)
        unique_bag_labels = np.unique(sequence_labels_np)  # TODO Get unique .bag file labels / input # !always returns the unique elements in sorted order
        
        bag_to_sequences = {}  # Map .bag file labels to list of sequence indices
        
        # Step 2: group indices by bag label
        for label in unique_bag_labels:
            # Find the indices where the label matches
            indices = np.where(sequence_labels_np == label)[0]  # Get indices of all sequences for the current bag
            bag_to_sequences[label] = list(indices)
        
        # Step 3: Create batches by randomly picking 1 sequences from the same .bag
        while bag_to_sequences:
            # Randomly select a bag from the available bags
            bag_label = random.choice(list(bag_to_sequences.keys()))
            sequence_indices = bag_to_sequences[bag_label]
            
            # Select 1 sequences randomly from this bag
            batch_indices = random.sample(sequence_indices, k=self.sequences_per_batch)
            
            # Remove the selected sequences from the bag
            for idx in batch_indices:
                sequence_indices.remove(idx)
            
            # If no more sequences in this bag, remove the bag from the pool
            if len(sequence_indices) == 0:
                del bag_to_sequences[bag_label]
            
            # Yield the indices for this batch
            yield batch_indices

    def __len__(self):
        # Total number of batches (each batch contains 1 sequence from one .bag)
        return len(self.dataset) // self.sequences_per_batch


class MultiBagBatchSampler(Sampler):
    def __init__(self, dataset, num_datasets=4, sequences_per_batch=1):
        """
        Sampler for loading multiple sequences from multiple datasets in parallel.

        Args:
            dataset: The dataset from which to sample.
            num_datasets: Number of different datasets to sample from in each batch.
            sequences_per_batch: Number of sequences to sample from each dataset in a batch.
        """
        self.dataset = dataset
        self.num_datasets = num_datasets
        self.sequences_per_batch = sequences_per_batch

    def __iter__(self):
        # Step 1: Convert labels for efficient comparison
        sequence_labels_np = np.array(self.dataset.sequence_labels)
        # unique_bag_labels_valid = np.unique(sequence_labels_np)  # Get unique .bag file labels (i.e., datasets)
        unique_bag_labels = self.dataset.unique_bag_labels
        
        bag_to_sequences = {}  # Map .bag file labels to a list of sequence indices

        # Step 2: Group indices by bag label (dataset)
        for label in unique_bag_labels:
            indices = np.where(sequence_labels_np == label)[0]  # Get indices of all sequences for the current dataset
            bag_to_sequences[label] = list(indices)

        # Step 3: Create batches by sampling from multiple datasets in parallel
        while bag_to_sequences:
            batch_indices = []  # Collect all sequences for the current batch
            selected_datasets = random.sample(list(bag_to_sequences.keys()), k=self.num_datasets)

            for dataset_label in selected_datasets:
                sequence_indices = bag_to_sequences[dataset_label]

                # Select `sequences_per_batch` sequences randomly from this dataset
                selected_sequences = random.sample(sequence_indices, k=self.sequences_per_batch)
                batch_indices.extend(selected_sequences)

                # Remove the selected sequences from the dataset
                for idx in selected_sequences:
                    sequence_indices.remove(idx)

                # If no more sequences in this dataset, remove it from the pool
                if len(sequence_indices) == 0:
                    del bag_to_sequences[dataset_label]

            # Ensure that the batch contains sequences from all selected datasets
            yield batch_indices

    def __len__(self):
        # Total number of batches (each batch contains sequences from `num_datasets`)
        return len(self.dataset) // (self.num_datasets * self.sequences_per_batch)


# # -- New college dataset
# NC_point_cloud_topic = "/os_cloud_node/points"
# nce_bag_path = 'data/Newer_College_Dataset/2021-07-01-10-37-38-quad-easy.bag'
# ncm_bag_path = 'data/Newer_College_Dataset/medium/2021-07-01-11-31-35_0-quad-medium.bag'
# nce_gt_pose_file = 'data/Newer_College_Dataset/gt-nc-quad-easy_TMU.csv'
# ncm_gt_pose_file = 'data/Newer_College_Dataset/medium/gt-nc-quad-medium.csv'

# # -- ASL
# ASL_point_cloud_topic = "/ouster/points"
# field_bag_path = './data/ASL/field_s/2023-08-09-19-05-05-field_s.bag'
# katzensee_bag_path = 'data/ASL/katzensee/2023-08-21-10-20-22-katzensee_s.bag'
# katzensees_gt_pose_file = 'data/ASL/katzensee/gt-katzensee_s.csv'
# fields_gt_pose_file = 'data/ASL/field_s/gt-field_s.csv'

# # list
# ts_field_name = "t"
# sequence_paths = [f'{field_bag_path}', f'{katzensee_bag_path}', f'{nce_bag_path}', f'{ncm_bag_path}']
# point_cloud_topics = [ASL_point_cloud_topic, ASL_point_cloud_topic, NC_point_cloud_topic, NC_point_cloud_topic]
# gt_poses_files = [fields_gt_pose_file, katzensees_gt_pose_file, nce_gt_pose_file, ncm_gt_pose_file]


# # Create dataset with sequences from all bags
# dataset = PointCloudDataset(sequence_paths, gt_poses_files, point_cloud_topics, num_sequences_per_bag=4) # 8/10

# # Use custom batch sampler for one or two sequences from the same bag per batch
# batch_sampler = SingleBagBatchSampler(dataset, sequences_per_batch=2)

# # Dataloader with custom batch sampler
# dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

# for batch in dataloader:
#     # 'batch' will be a list of point clouds (one list for each .bag file) 
#     print('a batch successfully loaded')