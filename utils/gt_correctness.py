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



# --------------------------------------------------------------
# correct ts

def replace_first_column(file1_path, file2_path, output_file_path):
    # Read the contents of both files
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        file1_lines = file1.readlines()
        file2_lines = file2.readlines()

    # Replace the first column of file1 with the first column of file2
    new_lines = []
    for line1, line2 in zip(file1_lines, file2_lines):
        line1_values = line1.split()
        line2_values = line2.split()

        # Replace the first value of line1 with the first value of line2
        line1_values[0] = line2_values[0]

        # Reconstruct the line and add to the new lines list
        new_lines.append(' '.join(line1_values) + '\n')

    # Write the modified content to the output file
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(new_lines)

# Example usage

# Path to the pin-slam copy file
file1_path = '/home/zjw/master_thesis/pin-slam-original/PIN_SLAM/experiments/mine_easy_original_pinslam_test_ncd_128_easy_rosbag__2024-11-09_14-11-18/odom_poses_tum.txt'  
# Path to the our file
file2_path = 'experiments/mine_e-wopretrain-400_40-1e-3---test_ncd_128_2024-11-07_10-59-51/slam_poses__tum.txt'  
# Path to the output file where the result will be saved
output_file_path = '/home/zjw/master_thesis/pin-slam-original/PIN_SLAM/experiments/mine_easy_original_pinslam_test_ncd_128_easy_rosbag__2024-11-09_14-11-18/odom_poses_tum_correctedts.txt'  

replace_first_column(file1_path, file2_path, output_file_path)