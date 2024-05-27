import os 
import numpy as np
import matplotlib.pyplot as plt

all_acc = []
all_gyro = []
for idx in range(1500):
    raw_imu_filename = os.path.join('./data/kitti_360/2013_05_28_drive_0000/data_poses/oxts_raw', 'data', f'{idx:010d}.txt') #w-17:20, a-11:14
    raw_imu = np.loadtxt(raw_imu_filename)
    lin_acc = raw_imu[11:14]
    ang_vel = raw_imu[17:20]
    all_acc.append(lin_acc)
    all_gyro.append(ang_vel)

all_acc = np.array(all_acc)
all_gyro = np.array(all_gyro)

accBias = np.array([ 0.95181334, -0.01282453, -0.04769801])
gyroBias = np.array([ 2.16058228e-04,  2.09481966e-04, -1.50556073e-05])
acc_sigma = np.array([0.01415831, 0.02812515, 0.01154988])
gyro_sigma = np.array([0.00108005, 0.00063807, 0.00116715])

gyro_data = all_gyro
acc_data = all_acc
indices = np.arange(1500)
highlight_indices = [41, 1085]

# gyroBias = np.mean(gyro_data, axis=0)
# accBias = np.mean(self.imu_curinter['acc'], axis=0) - np.array([0, 0, 9.8])
# gyro_sigma = (gyro_data.max(axis=0) - gyro_data.min(axis=0)) / 2
# acc_sigma = (acc_data.max(axis=0) - acc_data.min(axis=0)) / 2
# Plotting the IMU data
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# Gyroscope data
axs[0].plot(indices, gyro_data[:, 0], label='Gyro X', color='r')
axs[0].plot(indices, gyro_data[:, 1], label='Gyro Y', color='g')
axs[0].plot(indices, gyro_data[:, 2], label='Gyro Z', color='b')
axs[0].axhline(gyroBias[0], color='r', linestyle='--', label='Bias X')
axs[0].axhline(gyroBias[1], color='g', linestyle='--', label='Bias Y')
axs[0].axhline(gyroBias[2], color='b', linestyle='--', label='Bias Z')
axs[0].set_title('Gyroscope Data')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Angular Velocity [rad/s]')
axs[0].legend()
axs[0].grid(True)
# Display gyro sigmas on the plot
axs[0].text(0.05, 0.95, f'Gyro Bias X: {gyroBias[0]:.6f}\nGyro Bias Y: {gyroBias[1]:.6f}\nGyro Bias Z: {gyroBias[2]:.6f}\n'
                        f'Gyro Sigma X: {gyro_sigma[0]:.6f}\nGyro Sigma Y: {gyro_sigma[1]:.6f}\nGyro Sigma Z: {gyro_sigma[2]:.6f}',
            transform=axs[0].transAxes, fontsize=10, verticalalignment='top')
# Highlighting specific indices
axs[0].scatter(highlight_indices, gyro_data[highlight_indices, 0], color='r', s=100, zorder=5)
axs[0].scatter(highlight_indices, gyro_data[highlight_indices, 1], color='g', s=100, zorder=5)
axs[0].scatter(highlight_indices, gyro_data[highlight_indices, 2], color='b', s=100, zorder=5)


# Accelerometer data
axs[1].plot(indices, acc_data[:, 0], label='Accel X', color='r')
axs[1].plot(indices, acc_data[:, 1], label='Accel Y', color='g')
axs[1].plot(indices, acc_data[:, 2], label='Accel Z', color='b')
axs[1].axhline(accBias[0], color='r', linestyle='--', label='Bias X + Gravity')
axs[1].axhline(accBias[1], color='g', linestyle='--', label='Bias Y')
axs[1].axhline(accBias[2] + 9.8, color='b', linestyle='--', label='Bias Z')
axs[1].set_title('Accelerometer Data')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Linear Acceleration [m/s^2]')
axs[1].legend()
axs[1].grid(True)
# Display accel sigmas on the plot
axs[1].text(0.05, 0.95, f'Accel Bias X: {accBias[0]:.6f}\nAccel Bias Y: {accBias[1]:.6f}\nAccel Bias Z: {accBias[2]:.6f}\n'
                        f'Accel Sigma X: {acc_sigma[0]:.6f}\nAccel Sigma Y: {acc_sigma[1]:.6f}\nAccel Sigma Z: {acc_sigma[2]:.6f}',
            transform=axs[1].transAxes, fontsize=10, verticalalignment='top')

# Highlighting specific indices
axs[1].scatter(highlight_indices, acc_data[highlight_indices, 0], color='r', s=100, zorder=5)
axs[1].scatter(highlight_indices, acc_data[highlight_indices, 1], color='g', s=100, zorder=5)
axs[1].scatter(highlight_indices, acc_data[highlight_indices, 2], color='b', s=100, zorder=5)

plt.tight_layout()
plt.show()