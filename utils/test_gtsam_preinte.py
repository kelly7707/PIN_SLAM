import gtsam
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------test transforamtions ---------------------
# test transformation T_pose to velo
# ----------cam to Pose--------
# {
#     'image_00': array([[ 0.03717833, -0.09861821,  0.9944306 ,  1.5752681 ],
#        [ 0.99926756, -0.00535534, -0.03789026,  0.00439141],
#        [ 0.00906218,  0.99511093,  0.09834688, -0.65      ],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]]),  ?????that should be from pose to cam
#     'image_01': array([[ 0.01940009, -0.10515296,  0.99426681,  1.59772414],
#        [ 0.9997375 , -0.01008367, -0.02057327,  0.59814949],
#        [ 0.01218919,  0.99440493,  0.10492974, -0.64884331],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]]),
#     'image_02': array([[ 0.99951851,  0.00412766, -0.03075245,  0.72640369],
#        [-0.03079267,  0.01006084, -0.99947516, -0.14996585],
#        [-0.0038161 ,  0.99994087,  0.0101831 , -1.06864001],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]]),
#     'image_03': array([[-9.99682170e-01,  5.70340700e-04, -2.52038325e-02,
#          7.01684213e-01],
#        [-2.52033830e-02,  7.82081400e-04,  9.99682038e-01,
#          7.46365095e-01],
#        [ 5.89870900e-04,  9.99999531e-01, -7.67458300e-04,
#         -1.07519783e+00],
#        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#          1.00000000e+00]])
# }
# ----------cam to Lidar-------- [[ 0.04307104 -0.08829286  0.99516293  0.80439144]
#  [-0.99900437  0.00778461  0.04392797  0.29934896]
#  [-0.01162549 -0.99606414 -0.08786967 -0.17702258]
#  [ 0.          0.          0.          1.        ]]   ?????that should be from lidar to cam


# -------transforamtion everybody says------- [[ 0.99992906  0.0057743   0.01041756  0.77104934]
#  [ 0.00580536 -0.99997879 -0.00295331  0.29854144]
#  [ 0.01040029  0.00301357 -0.99994137 -0.83628022]
#  [ 0.          0.          0.          1.        ]]
# -------transforamtion I feel like------- [[-0.9967767  -0.00453018  0.080098    0.27036781]
#  [-0.01046623 -0.98252872 -0.18581651  0.54510788]
#  [ 0.07954037 -0.18605589  0.97931432 -0.73124351]
#  [ 0.          0.          0.          1.        ]]

def plot_frame(ax, R, t, frame_label, origin_color, colors, length=0.1):
    # Plot the origin
    ax.scatter(t[0], t[1], t[2], color=origin_color, label=frame_label)
    
    # Plot the x, y, z axes with different colors
    for i in range(3):
        ax.quiver(t[0], t[1], t[2], R[0, i], R[1, i], R[2, i], color=colors[i], length=length)
    
    # Add text labels for the axes
    ax.text(t[0] + R[0, 0] * length, t[1] + R[1, 0] * length, t[2] + R[2, 0] * length, 'X', color=colors[0])
    ax.text(t[0] + R[0, 1] * length, t[1] + R[1, 1] * length, t[2] + R[2, 1] * length, 'Y', color=colors[1])
    ax.text(t[0] + R[0, 2] * length, t[1] + R[1, 2] * length, t[2] + R[2, 2] * length, 'Z', color=colors[2])

T_camtoimu = np.array([[ 0.03717833, -0.09861821,  0.9944306 ,  1.5752681 ],
       [ 0.99926756, -0.00535534, -0.03789026,  0.00439141],
       [ 0.00906218,  0.99511093,  0.09834688, -0.65      ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

T_camtolidar = np.array([[ 0.04307104, -0.08829286 , 0.99516293  ,0.80439144],
        [-0.99900437 , 0.00778461 , 0.04392797 , 0.29934896],
        [-0.01162549 ,-0.99606414 ,-0.08786967 ,-0.17702258],
        [ 0.          ,0.          ,0.         , 1.        ]] ) # indeed from lidar to cam0

T_posetolidar_right = np.array([[ 0.99992906,0.0057743 ,  0.01041756 , 0.77104934],
        [ 0.00580536, -0.99997879, -0.00295331 , 0.29854144],
        [ 0.01040029 , 0.00301357, -0.99994137, -0.83628022],
        [ 0.     ,     0.     ,     0.    ,      1.        ]])

T_posetolidar_myfakeright = np.array([[-0.9967767,-0.00453018, 0.080098  ,  0.27036781],
 [-0.01046623 ,-0.98252872, -0.18581651 , 0.54510788],
 [ 0.07954037 ,-0.18605589  ,0.97931432 ,-0.73124351],
 [ 0.  ,        0.    ,      0.      ,    1.        ]])

#
calibration_IMU_initial_pose = np.array([[ 0.00000000e+00,-9.99999137e-01,-1.31367804e-03,0.00000000e+00],
 [ 9.95280614e-01 ,1.27477536e-04 ,-9.70385609e-02 , 0.00000000e+00],
 [ 9.70386446e-02 ,-1.30747829e-03 , 9.95279756e-01 , 0.00000000e+00],
 [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])

# Rotation matrix for 90 degree rotation about Z-axis
R_z_90 = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Calculate new transformation matrix
calibration_IMU_initial_pose = R_z_90 @ calibration_IMU_initial_pose



T=calibration_IMU_initial_pose
# Extract rotation and translation components
R = T[:3, :3]
t = T[:3, 3]

# Create a figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the original frame (identity matrix)
I_righthand = np.eye(3)
I_foward_left_down =np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
I=I_righthand
origin = np.array([0, 0, 0])
plot_frame(ax, I, origin, 'Original Frame', 'r', ['r', 'g', 'b'])


# Plot the transformed frame
plot_frame(ax, R, t, 'Transformed Frame', 'k', ['r', 'g', 'b'])


# T=np.linalg.inv(T_posetolidar_myfakeright)
# # Extract rotation and translation components
# R = T[:3, :3]
# t = T[:3, 3]
# plot_frame(ax, R, t, 'Fake right Transformed Frame', 'g', ['r', 'g', 'b'])

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()




# ------------------------------------------------------
# ---               test preintegration               --
# ------------------------------------------------------
def imu_calibration( imu_calibration_steps=30):
    num_samples = 0
    gyro_avg = np.zeros(3)
    accel_avg = np.zeros(3)

    ang_vel_list = []
    lin_acc_list = []
    timestamps = []

    for idx in range(imu_calibration_steps):
        num_samples += 1

        raw_imu_filename = os.path.join("./data/kitti_360/2013_05_28_drive_0000/data_poses/oxts_raw", 'data', f'{idx:010d}.txt') #w-17:20, a-11:14
        raw_imu = np.loadtxt(raw_imu_filename)
        lin_acc = raw_imu[11:14]
        ang_vel = raw_imu[17:20]

GRAVITY = 9.81 
params = gtsam.PreintegrationCombinedParams.MakeSharedU(GRAVITY)

T_posetolidar_right = np.array([[ 0.99992906,0.0057743 ,  0.01041756 , 0.77104934],
        [ 0.00580536, -0.99997879, -0.00295331 , 0.29854144],
        [ 0.01040029 , 0.00301357, -0.99994137, -0.83628022],
        [ 0.     ,     0.     ,     0.    ,      1.        ]])
velocity = np.array([0,0,0])
past_pose = np.eye(4)
# for frame_id in range(150):
#     read_imu
#     last_pose_imu = past_pose @ np.linalg.inv(T_posetolidar_right)