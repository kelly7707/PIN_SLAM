""" convert gt wrt. base frame to wrt. lidar frame"""
import yaml
import numpy as np
import csv
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from numpy.linalg import svd

# -- or import fromm slam_dataset.py
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

# Read TUM format poses
def read_tum_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            timestamp = float(row[0])
            translation = np.array([float(row[1]), float(row[2]), float(row[3])])
            quaternion = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
            poses.append((timestamp, translation, quaternion))
    return poses

# Write poses to a new TUM file with precise formatting for timestamp, translation, and quaternion
def write_tum_poses(file_path, poses):
    with open(file_path, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for timestamp, translation, quaternion in poses:
            # Format the timestamp, translation, and quaternion to keep full precision
            writer.writerow([f'{timestamp:.18e}'] +
                            [f'{x:.18e}' for x in translation] +
                            [f'{x:.18e}' for x in quaternion])

# Function to orthogonalize a rotation matrix using SVD
def orthogonalize_matrix(matrix):
    U, _, Vt = svd(matrix)
    return np.dot(U, Vt)
            
# Apply a transformation to the pose
def transform_pose(translation, quaternion, T_L_GT):
    # Create the pose homogeneous matrix
    T_WL_L = create_homogeneous_transform(translation, quaternion)
    
    # Apply the transformation
    T_Wl_GT = T_WL_L @ T_L_GT
    
    # Extract the new translation and rotation
    new_translation = T_Wl_GT[:3, 3]
    new_rotation_matrix = T_Wl_GT[:3, :3]

    # Orthogonalize the rotation matrix to ensure it's valid
    new_rotation_matrix = orthogonalize_matrix(new_rotation_matrix)
    
    new_quaternion = Quaternion(matrix=new_rotation_matrix)
    
    return new_translation, [new_quaternion.x, new_quaternion.y, new_quaternion.z, new_quaternion.w]


# Function to plot 3D poses
def plot_3d_poses(poses, transformed_poses):
    # Extract positions before and after transformation
    original_positions = np.array([pose[1] for pose in poses])  # Original translations
    transformed_positions = np.array([pose[1] for pose in transformed_poses])  # Transformed translations

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original positions
    ax.plot(original_positions[:, 0], original_positions[:, 1], original_positions[:, 2], 'r-', label='Original Poses')

    # Plot transformed positions
    ax.plot(transformed_positions[:, 0], transformed_positions[:, 1], transformed_positions[:, 2], 'b-', label='Transformed Poses')

    # Adding labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title and legend
    ax.set_title('3D Poses Before and After Transformation')
    ax.legend()

    # Show the plot
    plt.show()

# ----------- IMU Lidar (can be used for both new college and asl dataset)--------------
# Extract translation and rotation for os_sensor_to_os_imu
calib_file_path = 'data/Newer_College_Dataset/os_imu_lidar_transforms.yaml'
with open(calib_file_path, 'r') as file:
    calibration_data = yaml.safe_load(file)

# Extract translation and rotation for os_sensor_to_os_imu
os_sensor_to_os_imu_data = calibration_data['os_sensor_to_os_imu']
translation_imu = np.array(os_sensor_to_os_imu_data['translation'])
rotation_imu = np.array(os_sensor_to_os_imu_data['rotation'])

T_I_L = create_homogeneous_transform(translation_imu, rotation_imu)
T_L_I = np.linalg.inv(T_I_L)


Flag = False # True for New College, False for ASL
if Flag:
    print('new college')
    calib_file_path = 'data/Newer_College_Dataset/os_imu_lidar_transforms.yaml'
    with open(calib_file_path, 'r') as file:
        calibration_data = yaml.safe_load(file)
    os_sensor_to_base_data = calibration_data['os_sensor_to_base']
    translation_base = np.array(os_sensor_to_base_data['translation'])
    rotation_base = np.array(os_sensor_to_base_data['rotation'])

    T_GT_L = create_homogeneous_transform(translation_base, rotation_base)
    T_L_GT = np.linalg.inv(T_GT_L)

    
    # traj_ref_file = 'data/Newer_College_Dataset/gt-nc-quad-easy_TMU.csv'
    # output_file = 'data/Newer_College_Dataset/transformed_gt-nc-quad-easy_TMU.csv'
    # traj_ref_file = 'data/Newer_College_Dataset/medium/gt-nc-quad-medium.csv'
    # output_file = 'data/Newer_College_Dataset/transformed_gt-nc-quad-medium.csv'

else:
    print('ASL')    
    # -----------ASL dataset --------------
    """# Rotations are in quaternion (qx, qy, qz, qw) 
    prism_to_os_imu: # T_imu_prism
    translation : [-0.006253, 0.011775, 0.10825] 
    rotation : [0.0, 0.0, 0.0, 1.0]"""
    translation_imu_prism = np.array([-0.006253, 0.011775, 0.10825])
    rotation_imu_prism = np.array([0.0, 0.0, 0.0, 1.0])
    T_I_prism = create_homogeneous_transform(translation_imu_prism, rotation_imu_prism)

    T_L_GT = T_L_I @ T_I_prism
    # T_GT_L = np.linalg.inv(T_L_GT)

    # # tum format poses 'timestamp tx ty tz qx qy qz qw'
    # traj_ref_file = 'data/ASL/katzensee/gt-katzensee_s.csv' # ground truth in tum format
    # output_file = 'data/ASL/katzensee/transformed_gt-katzensee_s.csv'

# ----------- [wrong! esp. gt has only position info] transform gt wrt. base frame to wrt. lidar frame 
# poses = read_tum_poses(traj_ref_file)
# ----------- transform estimated poses instead
traj_est_file = 'experiments/history/pretrained/ncm-kitti360 4-5s/pinslam-katzensee_d - test_ros_2024-10-31_13-43-09/slam_poses__tum_correctedts.txt'

output_file = traj_est_file + '_transformed2GT'
poses = read_tum_poses(traj_est_file)

# Apply the transformation to all poses
transformed_poses = []
for timestamp, translation, quaternion in poses:
    new_translation, new_quaternion = transform_pose(translation, quaternion, T_L_GT)
    transformed_poses.append((timestamp, new_translation, new_quaternion))

# # Plot the poses
# plot_3d_poses(poses, transformed_poses)

# Save the transformed poses into a new TUM file
write_tum_poses(output_file, transformed_poses)