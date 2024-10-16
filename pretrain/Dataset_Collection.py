"""Description: This script is used to extract multiple (5s or 50 msgs) sequences from each .bag file, mixing them, 
and using one or two (n) sequences from a single bag per training iteration"""

import rosbag
import random
import numpy as np
import torch
from sensor_msgs import point_cloud2
from torch.utils.data import DataLoader, Dataset, Sampler


def load_multiple_sequences(bag_file_path, point_cloud_topic, num_sequences=5, num_msgs=50):
    """Extract multiple random, non-overlapping sequences by selecting 50 consecutive point cloud messages."""
    
    # Step 1: Preload all message timestamps
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
        with rosbag.Bag(bag_file_path, 'r') as bag:
            start_time = timestamps[start_idx]
            end_time = timestamps[start_idx + num_msgs - 1]

            # Read messages within the selected time interval
            for topic, msg, t in bag.read_messages(topics=[point_cloud_topic], start_time=start_time, end_time=end_time):
                points = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
                point_coords.append(points)

        # Concatenate the point clouds for the current sequence into a tensor
        if point_coords:
            point_coords = np.concatenate(point_coords, axis=0)
            selected_sequences.append(torch.tensor(point_coords, dtype=torch.float32))

    return selected_sequences  # List of point cloud tensors (one per sequence)

# class PointCloudDataset(Dataset):
#     def __init__(self, sequence_paths, point_cloud_topics, num_sequences_per_batch=5):
#         self.sequences = sequence_paths  # List of .bag file paths
#         self.topics = point_cloud_topics
#         self.num_sequences_per_batch = num_sequences_per_batch
    
#     def __len__(self):
#         return len(self.sequences)
    
#     def __getitem__(self, idx):
#         # Load multiple 5-second sequences from the selected .bag file
#         sequence_path = self.sequences[idx]
#         point_cloud_topic = self.topics[idx]
#         point_clouds = load_multiple_sequences(sequence_path, point_cloud_topic, num_sequences=self.num_sequences_per_batch)
#         return point_clouds  # Return a list of point clouds (sequences)

class PointCloudDataset(Dataset):
    def __init__(self, sequence_paths, point_cloud_topics, num_sequences_per_bag=10):
        self.sequences = []  # Store all sequences, but track which bag they came from
        self.sequence_labels = []  # Keep track of which .bag each sequence came from
        
        # Load multiple sequences from each bag and store them
        for sequence_path, point_cloud_topic in zip(sequence_paths, point_cloud_topics):
            print('--------loading sequences from bag')
            sequences_from_bag = load_multiple_sequences(sequence_path, point_cloud_topic, num_sequences=num_sequences_per_bag)
            self.sequences.extend(sequences_from_bag)  # Store sequences from the bag
            self.sequence_labels.extend([sequence_path] * num_sequences_per_bag)  # Track source of sequences
        print('--------sequences loaded for all bags')

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Return a sequence and the label (the bag it came from)
        return self.sequences[idx], self.sequence_labels[idx]

class SingleBagBatchSampler(Sampler):
    def __init__(self, dataset, sequences_per_batch=2):
        self.dataset = dataset
        self.sequences_per_batch = sequences_per_batch

    def __iter__(self):
        # Step 1: Convert for efficient comparison
        sequence_labels_np = np.array(self.dataset.sequence_labels)
        unique_bag_labels = np.unique(sequence_labels_np)  # Get unique .bag file labels / input

        bag_to_sequences = {}  # Map .bag file labels to list of sequence indices
        
        # Step 2: group indices by bag label
        for label in unique_bag_labels:
            # Find the indices where the label matches
            indices = np.where(sequence_labels_np == label)[0]  # Get indices of all sequences for the current bag
            bag_to_sequences[label] = list(indices)
        
        # Step 3: Create batches by randomly picking 1 or 2 sequences from the same .bag
        while bag_to_sequences:
            # Randomly select a bag from the available bags
            bag_label = random.choice(list(bag_to_sequences.keys()))
            sequence_indices = bag_to_sequences[bag_label]
            
            # Select 1 or 2 sequences randomly from this bag
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
        # Total number of batches (each batch is either 1 or 2 sequences from one .bag)
        return len(self.dataset) // self.sequences_per_batch



# -- New college dataset
NC_point_cloud_topic = "/os_cloud_node/points"
nce_bag_path = 'data/Newer_College_Dataset/2021-07-01-10-37-38-quad-easy.bag'
ncm_bag_path = 'data/Newer_College_Dataset/medium/2021-07-01-11-31-35_0-quad-medium.bag'

# -- ASL
ASL_point_cloud_topic = "/ouster/points"
field_bag_path = './data/ASL/field_s/2023-08-09-19-05-05-field_s.bag'
katzensee_bag_path = 'data/ASL/katzensee/2023-08-21-10-20-22-katzensee_s.bag'

# list
ts_field_name = "t"
sequence_paths = [f'{field_bag_path}', f'{katzensee_bag_path}', f'{nce_bag_path}', f'{ncm_bag_path}']
point_cloud_topics = [ASL_point_cloud_topic, ASL_point_cloud_topic, NC_point_cloud_topic, NC_point_cloud_topic]

# # Create dataset and dataloader, defining sequences per batch
# dataset = PointCloudDataset(sequence_paths, point_cloud_topics, num_sequences_per_batch=5)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Each batch contains multiple sequences

# for batch in dataloader:
#     # 'batch' will be a list of point clouds (one list for each .bag file) 
#     print('a batch successfully loaded')
#     for point_clouds in batch:
#         for sequence in point_clouds:
#             print(sequence.shape)  # Each sequence is an Nx3 tensor of point cloud coordinates


# Create dataset with sequences from all bags
dataset = PointCloudDataset(sequence_paths, point_cloud_topics, num_sequences_per_bag=4) # 8/10

# Use custom batch sampler for one or two sequences from the same bag per batch
batch_sampler = SingleBagBatchSampler(dataset, sequences_per_batch=2)

# Dataloader with custom batch sampler
dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

for batch in dataloader:
    # 'batch' will be a list of point clouds (one list for each .bag file) 
    print('a batch successfully loaded')