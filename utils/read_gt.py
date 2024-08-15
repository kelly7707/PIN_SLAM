import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R

# Function to read TUM format CSV file
def read_tum_csv(file_path):
    # Load the CSV file using pandas
    df = pd.read_csv(file_path, header=None, delim_whitespace=True)
    
    # Assign column names for clarity
    df.columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    
    return df

# Function to plot the GT poses in 3D
def plot_gt_trajectory_3d(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    ax.plot(df['tx'], df['ty'], df['tz'], label='Ground Truth Trajectory')
    
    # Mark the starting point
    ax.scatter(df['tx'][0], df['ty'][0], df['tz'][0], color='red', s=100, label='Start Point')

    # Calculate the direction vector from the quaternion
    start_orientation = R.from_quat([df['qx'][0], df['qy'][0], df['qz'][0], df['qw'][0]])
    direction_vector = start_orientation.apply([1, 0, 0])  # Assuming forward direction is along x-axis

    # Plot the starting orientation as an arrow
    ax.quiver(df['tx'][0], df['ty'][0], df['tz'][0], 
              direction_vector[0], direction_vector[1], direction_vector[2], 
              color='blue', length=1.0, normalize=True, label='Start Orientation')

    # Labels and title
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Ground Truth Trajectory (3D)')
    
    plt.legend()
    plt.show()

# Function to plot the GT poses in 2D (x, y)
def plot_gt_trajectory_2d(df):
    plt.figure()
    
    # Plot the 2D trajectory
    plt.plot(df['tx'], df['ty'], label='Ground Truth Trajectory (2D)')
    
    # Mark the starting point
    plt.scatter(df['tx'][0], df['ty'][0], color='red', s=100, label='Start Point')

    # Calculate the direction vector from the quaternion
    start_orientation = R.from_quat([df['qx'][0], df['qy'][0], df['qz'][0], df['qw'][0]])
    direction_vector = start_orientation.apply([1, 0, 0])  # Assuming forward direction is along x-axis

    # Plot the starting orientation as an arrow
    plt.quiver(df['tx'][0], df['ty'][0], 
               direction_vector[0], direction_vector[1], 
               color='blue', scale=10, label='Start Orientation')

    # Labels and title
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Ground Truth Trajectory (2D)')
    
    plt.legend()
    plt.axis('equal')  # Ensure equal scaling of x and y axes
    plt.show()



if __name__ == "__main__":
    # Path to your TUM format CSV file
    # file_path = 'data/ASL/field_s/gt-field_s.csv'
    file_path = 'data/ASL/katzensee/gt-katzensee_s.csv'
    
    # Read the CSV file
    df = read_tum_csv(file_path)
    
    # Plot the GT trajectory
    plot_gt_trajectory_2d(df)
