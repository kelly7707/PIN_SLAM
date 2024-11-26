# # Let's load the CSV file to examine its content and perform the required operation.
# import pandas as pd

# # Load the file to inspect its content
# file_path = 'data/Newer_College_Dataset/mine_easy/easy_gt_state_tum.csv'
# with open(file_path, 'r') as file:
#     file_content = file.readlines()

# # Process each line to replace the first space with a period
# modified_content = [line.replace(' ', '.', 1) for line in file_content]

# # Write the modified content to a new CSV file
# output_path = 'data/Newer_College_Dataset/mine_easy/easy_gt_state_tum.csv'
# with open(output_path, 'w') as file:
#     file.writelines(modified_content)

# output_path



# # --------------------------------------------------------------
# # Load the text file and check for missing numbers in the first column

# # Read the file
# file_path = 'experiments/cloister_all-pin_imu-ba0-test_ncd_128_2024-11-09_00-27-50/slam_poses__tum.txt'  # Replace with the path to your file
# with open(file_path, 'r') as file:
#     lines = file.readlines()

# # Extract the first column as a list of floats
# first_column = [float(line.split()[0]) for line in lines]

# # Check for missing numbers in the first column
# missing_numbers = []
# for i in range(int(first_column[0]), int(first_column[-1])):
#     if i not in first_column:
#         missing_numbers.append(i)

# # Output the missing numbers
# if missing_numbers:
#     print("Missing numbers in the first column:", missing_numbers)
# else:
#     print("No missing numbers in the first column.")



# # --------------------------------------------------------------
# # correct ts

# def replace_first_column(file1_path, file2_path, output_file_path):
#     # Read the contents of both files
#     with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
#         file1_lines = file1.readlines()
#         file2_lines = file2.readlines()

#     # Replace the first column of file1 with the first column of file2
#     new_lines = []
#     for line1, line2 in zip(file1_lines, file2_lines):
#         line1_values = line1.split()
#         line2_values = line2.split()

#         # Replace the first value of line1 with the first value of line2
#         line1_values[0] = line2_values[0]

#         # Reconstruct the line and add to the new lines list
#         new_lines.append(' '.join(line1_values) + '\n')

#     # Write the modified content to the output file
#     with open(output_file_path, 'w') as output_file:
#         output_file.writelines(new_lines)

# # Example usage

# # Path to the pin-slam copy file
# file1_path = '/home/zjw/master_thesis/pin-slam-original/PIN_SLAM/experiments/mine_easy_original_pinslam_test_ncd_128_easy_rosbag__2024-11-09_14-11-18/odom_poses_tum.txt'  
# # Path to the our file
# file2_path = 'experiments/mine_e-wopretrain-400_40-1e-3---test_ncd_128_2024-11-07_10-59-51/slam_poses__tum.txt'  
# # Path to the output file where the result will be saved
# output_file_path = '/home/zjw/master_thesis/pin-slam-original/PIN_SLAM/experiments/mine_easy_original_pinslam_test_ncd_128_easy_rosbag__2024-11-09_14-11-18/odom_poses_tum_correctedts.txt'  

# replace_first_column(file1_path, file2_path, output_file_path)



# # --------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_timing_detail(time_table: np.ndarray, saving_path: str, with_loop=False):
    frame_count = time_table.shape[0]
    time_table_ms = time_table * 1e3

    for i in range(time_table.shape[1] - 1):  # accumulated time
        time_table_ms[:, i + 1] += time_table_ms[:, i]

    font2 = {'weight': 'normal', 'size': 18}

    color_values = np.linspace(0, 1, 6)
    colors = plt.cm.viridis(color_values)  # Use the viridis colormap

    fig = plt.figure(figsize=(12.0, 4.0))

    frame_array = np.arange(frame_count)
    realtime_limit = 100.0 * np.ones([frame_count, 1])
    ax1 = fig.add_subplot(111)

    line_width_1 = 0.6
    line_width_2 = 1.0
    alpha_value = 1.0

    ax1.fill_between(frame_array, time_table_ms[:, 0], facecolor=colors[0], edgecolor='face', where=time_table_ms[:, 0] > 0, alpha=alpha_value, interpolate=True)
    ax1.fill_between(frame_array, time_table_ms[:, 0], time_table_ms[:, 1], facecolor=colors[1], edgecolor='face', where=time_table_ms[:, 1] > time_table_ms[:, 0], alpha=alpha_value, interpolate=True)
    ax1.fill_between(frame_array, time_table_ms[:, 1], time_table_ms[:, 2], facecolor=colors[2], edgecolor='face', where=time_table_ms[:, 2] > time_table_ms[:, 1], alpha=alpha_value, interpolate=True)
    ax1.fill_between(frame_array, time_table_ms[:, 2], time_table_ms[:, 3], facecolor=colors[3], edgecolor='face', where=time_table_ms[:, 3] > time_table_ms[:, 2], alpha=alpha_value, interpolate=True)
    if with_loop:
        ax1.fill_between(frame_array, time_table_ms[:, 3], time_table_ms[:, 4], facecolor=colors[4], edgecolor='face', where=time_table_ms[:, 4] > time_table_ms[:, 3], alpha=alpha_value, interpolate=True)

    ax1.plot(frame_array, realtime_limit, "--", linewidth=line_width_2, color='k')

    plt.tick_params(labelsize=12)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()

    plt.xlim((0, frame_count - 1))
    plt.ylim((0, 1400))

    plt.xlabel('Frame ID', font2)
    plt.ylabel('Runtime (ms)', font2)
    plt.tight_layout()

    if with_loop:
        legend = plt.legend(('Pre-processing', 'Odometry', "Mapping preparation", "Map optimization", "Loop closures"), prop=font2, loc=2)
    else:
        legend = plt.legend(('Pre-processing', 'Odometry', "Mapping preparation", "Map optimization"), prop=font2, loc=2)

    plt.savefig(saving_path, dpi=500)
    plt.close(fig)  # Close the figure to free memory

# Load the time_table from the .npy file
table_file = 'experiments/history/final_pre/ours-pretrian/nce- pretrain ncm_kitti230 - nc config - test_ncd_128_2024-11-06_09-03-43/time_table.npy'
time_table = np.load(table_file)

# Define the saving path
run_path = 'experiments/history/final_pre/ours-pretrian/nce- pretrain ncm_kitti230 - nc config - test_ncd_128_2024-11-06_09-03-43'
saving_path = os.path.join(run_path, "time_details_corrected.png")

# Plot the timing details
plot_timing_detail(time_table, saving_path, with_loop=False)