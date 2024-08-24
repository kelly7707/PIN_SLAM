"""
# read the gt poses in TUM format 
gt_pose_file = 'data/Newer_College_Dataset/gt-nc-quad-easy_TMU.csv'
# read the estimated poses, also in TUM format
estimated_pose_file = 'experiments/history/!new college norm 800warmup 0.2dropout test_ros_2024-08-08_09-07-45/odom_poses__tum.txt'
# calculate the Root Mean Square Error (RMSE) for translation errors and the Mean Rotation Error (MRE) for rotation errors. 
# visualize the error in boxplot
 """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def read_tum_poses(file_path):
    """
    Reads TUM format pose files.
    Returns: numpy array of timestamps, translations, and rotations.
    """
    data = pd.read_csv(file_path, sep=' ', header=None)
    timestamps = data[0].values
    translations = data.iloc[:, 1:4].values
    rotations = data.iloc[:, 4:8].values  # Quaternion format (x, y, z, w)
    return timestamps, translations, rotations

def align_first_frame(gt_trans, gt_rot, est_trans, est_rot):
    """
    Align the first frame of the estimated poses to the ground truth poses.
    """
    # Compute the translation and rotation difference for the first frame
    trans_diff = gt_trans[0] - est_trans[0]
    gt_rot_first = R.from_quat(gt_rot[0])
    est_rot_first = R.from_quat(est_rot[0])
    rot_diff = gt_rot_first * est_rot_first.inv()

    # Apply the translation and rotation difference to all estimated poses
    aligned_trans = est_trans + trans_diff
    aligned_rot = rot_diff * R.from_quat(est_rot)

    return aligned_trans, aligned_rot.as_quat()

def compute_translation_rmse(gt_trans, est_trans):
    """
    Computes the Root Mean Square Error (RMSE) for translation errors.
    """
    translation_errors = np.linalg.norm(gt_trans - est_trans, axis=1)
    rmse = np.sqrt(np.mean(translation_errors ** 2))
    return rmse, translation_errors

def compute_rotation_mre(gt_rot, est_rot):
    """
    Computes the Mean Rotation Error (MRE) for rotation errors.
    """
    gt_rot = R.from_quat(gt_rot)
    est_rot = R.from_quat(est_rot)
    rotation_errors = gt_rot.inv() * est_rot  # Relative rotation
    angles = rotation_errors.magnitude()  # Rotation angle in radians
    mre = np.mean(angles)
    return mre, angles

def plot_errors(translation_errors, rotation_errors):
    """
    Visualizes translation and rotation errors using boxplots.
    """
    plt.figure(figsize=(12, 6))

    # Boxplot for translation errors
    plt.subplot(1, 2, 1)
    plt.boxplot(translation_errors)
    plt.title('Translation Errors (m)')
    plt.ylabel('Error (meters)')

    # Boxplot for rotation errors
    plt.subplot(1, 2, 2)
    plt.boxplot(rotation_errors)
    plt.title('Rotation Errors (radians)')
    plt.ylabel('Error (radians)')

    plt.tight_layout()
    plt.show()

# File paths
# new college dataset
gt_pose_file = 'data/Newer_College_Dataset/gt-nc-quad-easy_TMU.csv'
estimated_pose_file = 'experiments/history/!new college norm 800warmup 0.2dropout test_ros_2024-08-08_09-07-45/odom_poses__tum.txt'
# # katzensee dataset
# estimated_pose_file = 'experiments/history/asl dataset/katzensee pinslam corrected-poses saved test_ros_2024-08-19_10-39-09/slam_poses__tum.txt'
# gt_pose_file = 'data/ASL/katzensee/gt-katzensee_s.csv'


# Read poses
gt_timestamps, gt_translations, gt_rotations = read_tum_poses(gt_pose_file)
est_timestamps, est_translations, est_rotations = read_tum_poses(estimated_pose_file)

# Align the first frame of estimated poses to the ground truth
aligned_translations, aligned_rotations = align_first_frame(gt_translations, gt_rotations, est_translations, est_rotations)

# Compute errors
translation_rmse, translation_errors = compute_translation_rmse(gt_translations, aligned_translations)
rotation_mre, rotation_errors = compute_rotation_mre(gt_rotations, aligned_rotations)

# Print RMSE and MRE
print(f"Translation RMSE: {translation_rmse:.4f} meters")
print(f"Rotation MRE: {rotation_mre:.4f} radians")

# Plot errors
plot_errors(translation_errors, rotation_errors)
