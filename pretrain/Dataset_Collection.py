"""Description: This script is used to load multiple sequences from a rosbag file."""
# import rospy
import rosbag
import random
import numpy as np
import torch
from sensor_msgs import point_cloud2
from torch.utils.data import DataLoader


def load_multiple_sequences(bag_file_path, point_cloud_topic, num_sequences=5, sequence_duration=5):
    """extract multiple random, non-overlapping sequences by randomly sampling start times and slicing out 5-second windows from the dataset."""
    # Store all point cloud messages and their timestamps
    point_clouds = []
    timestamps = []
    
    # Open the rosbag and read messages 
    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[point_cloud_topic]):
            point_clouds.append(msg)
            timestamps.append(t.to_sec())
    
    # Ensure we have enough data for multiple 5-second sequences
    total_duration = timestamps[-1] - timestamps[0]
    if total_duration < sequence_duration * num_sequences:
        raise ValueError(f"Not enough data in {bag_file_path} for {num_sequences} sequences of {sequence_duration} seconds")

    timestamps_np = np.array(timestamps)
    point_clouds_np = np.array(point_clouds)
    # Randomly select start times for each 5-second sequence
    selected_sequences = []
    random.seed(42) # seed for reproducibility 

    for _ in range(num_sequences):
        start_time = random.uniform(timestamps[0], timestamps[-1] - sequence_duration)
        end_time = start_time + sequence_duration

        # Extract the point clouds within the selected 5-second window
        # selected_point_clouds = [pc for pc, ts in zip(point_clouds, timestamps) if start_time <= ts <= end_time]
        # -- method 1
        mask = (timestamps_np >= start_time) & (timestamps_np <= end_time)
        selected_point_clouds = [point_clouds[i] for i in range(len(point_clouds)) if mask[i]]

        point_coords = np.concatenate([np.array(list(point_cloud2.read_points(pc, field_names=("x", "y", "z"), skip_nans=True))) for pc in selected_point_clouds], axis=0)
        selected_sequences.append(torch.tensor(point_coords, dtype=torch.float32))

        # # # -- method 2 (depends on efficiency)
        # mask = (timestamps_np >= start_time) & (timestamps_np <= end_time)
        # selected_point_clouds = point_clouds_np[mask]
        # # Convert point clouds to a usable format (e.g., N x 3 numpy array for coordinates)
        # point_coords = np.concatenate([np.array(pc.points)[:, :3] for pc in selected_point_clouds], axis=0)
        # selected_sequences.append(torch.tensor(point_coords, dtype=torch.float32))

    return selected_sequences  # List of point cloud tensors (one per sequence)

# pydevd warning: Computing repr of point_clouds (list) was slow (took 13.93s)
# Customize report timeout by setting the `PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT` environment variable to a higher timeout (default is: 0.5s)


class PointCloudDataset(Dataset):
    def __init__(self, sequence_paths, point_cloud_topics, num_sequences_per_batch=5):
        self.sequences = sequence_paths  # List of .bag file paths
        self.topics = point_cloud_topics
        self.num_sequences_per_batch = num_sequences_per_batch
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Load multiple 5-second sequences from the selected .bag file
        sequence_path = self.sequences[idx]
        point_cloud_topic = self.topics[idx]
        point_clouds = load_multiple_sequences(sequence_path, point_cloud_topic, num_sequences=self.num_sequences_per_batch)
        return point_clouds  # Return a list of point clouds (sequences)

# -- New college dataset
NC_point_cloud_topic = "/os_cloud_node/points"
nce_bag_path = 'data/Newer_College_Dataset/2021-07-01-10-37-38-quad-easy.bag'
ncm_bag_path = 'data/Newer_College_Dataset/medium/2021-07-01-11-31-35_0-quad-medium.bag'

# -- ASL
ASL_point_cloud_topic = "/ouster/points"
ts_field_name = "t"
field_bag_path = './data/ASL/field_s/2023-08-09-19-05-05-field_s.bag'
katzensee_bag_path = 'data/ASL/katzensee/2023-08-21-10-20-22-katzensee_s.bag'

# list
sequence_paths = [f'{field_bag_path}', f'{katzensee_bag_path}', f'{nce_bag_path}', f'{ncm_bag_path}']
point_cloud_topics = [ASL_point_cloud_topic, ASL_point_cloud_topic, NC_point_cloud_topic, NC_point_cloud_topic]

# Create dataset and dataloader, defining sequences per batch
dataset = PointCloudDataset(sequence_paths, point_cloud_topics, num_sequences_per_batch=5)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Each batch contains multiple sequences

for batch in dataloader:
    # testing: 'batch' will be a list of point clouds (one list for each .bag file) 
    for point_clouds in batch:
        for sequence in point_clouds:
            print(sequence.shape)  # Each sequence is an Nx3 tensor of point cloud coordinates


# ----------- pretraing
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for batch in dataloader:
    # Each 'batch' contains multiple 5-second sequences from a .bag file
    for point_clouds in batch:
        optimizer.zero_grad()

        # Iterate over each 5-second sequence in the batch
        for point_coords in point_clouds:
            num_points = point_coords.shape[0]

            # Reinitialize geo_features for this sequence
            geo_features = initialize_geo_features(batch_size=1, num_points=num_points)

            # Forward pass through the decoder
            sdf_pred = decoder(point_coords, geo_features)

            # Assuming you have ground truth SDF values
            sdf_gt = get_sdf_ground_truth(point_coords)
            loss = criterion(sdf_pred, sdf_gt)

            # Backward pass and optimization
            loss.backward()

        # Update weights after accumulating gradients for all sequences in the batch
        optimizer.step()

    print(f"Batch Loss: {loss.item()}")