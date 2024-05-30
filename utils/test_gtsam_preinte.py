import gtsam
import numpy as np
import os

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
