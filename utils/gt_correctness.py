# Let's load the CSV file to examine its content and perform the required operation.
import pandas as pd

# Load the file to inspect its content
file_path = 'data/Newer_College_Dataset/mine_easy/easy_gt_state_tum.csv'
with open(file_path, 'r') as file:
    file_content = file.readlines()

# Process each line to replace the first space with a period
modified_content = [line.replace(' ', '.', 1) for line in file_content]

# Write the modified content to a new CSV file
output_path = 'data/Newer_College_Dataset/mine_easy/easy_gt_state_tum.csv'
with open(output_path, 'w') as file:
    file.writelines(modified_content)

output_path