"""figure out why the imu trajectory is fliped"""
import gtsam
import numpy as np
import matplotlib.pyplot as plt

class imu_pre:
    def imu_Preintegration_init(self,dataset):
        # source gtsam.CombinedImuFactorExample.py
        self.GRAVITY = 9.81 
        params = gtsam.PreintegrationCombinedParams.MakeSharedU(self.GRAVITY) # Dwrong!OXTS coordinates are defined as x = forward, y = right, z = down// see imu dataformat: forward,left,top
        
        #self.imu_calib_initial_pose
        self.T_Wi_I0, self.accBias, self.gyroBias, self.accel_sigma, self.gyro_sigma = self.imu_calibration(dataset)
        self.imu_bias = gtsam.imuBias.ConstantBias(self.accBias, self.gyroBias)

        I_3x3 = np.eye(3)
        params.setGyroscopeCovariance(self.gyro_sigma**2 * I_3x3)
        params.setAccelerometerCovariance(self.accel_sigma**2 * I_3x3)
        params.setIntegrationCovariance(1e-6 * I_3x3)  # 1e-3**2 * I_3x3 # 1e-5 * I_3x3

        self.pim = gtsam.PreintegratedCombinedMeasurements(params, self.imu_bias)

        self.velocity = np.array([0, 0, 0]) # np.array([ 0.0005, -0.0007, -0.0058])# np.array([0, 0, 0])

    # --- IMU
    def preintegration(self, acc, gyro, dts, last_pose, cur_id: int):
        #
        for i,dt in enumerate(dts):
            # self.pim.integrate() # https://github.com/borglab/gtsam/blob/0fee5cb76e7a04b590ff0dc1051950da5b265340/python/gtsam/examples/PreintegrationExample.py#L159C16-L159C70 
            self.pim.integrateMeasurement(acc[i], gyro[i], dt) # https://github.com/borglab/gtsam/blob/4abef9248edc4c49943d8fd8a84c028deb486f4c/python/gtsam/examples/CombinedImuFactorExample.py#L175C12-L177C74 
        
        if cur_id == 1:
            initial_pose = gtsam.Pose3(self.T_Wi_I0) #self.imu_calib_initial_pose  # last_pose @ self.imu_calib_initial_pose  
        else:
            initial_pose = gtsam.Pose3(last_pose)
        initial_state = gtsam.NavState(
            initial_pose,
            self.velocity) # https://github.com/borglab/gtsam/blob/4abef9248edc4c49943d8fd8a84c028deb486f4c/python/gtsam/examples/CombinedImuFactorExample.py#L164C9-L168C46 

        imu_prediction = self.pim.predict(initial_state, self.imu_bias)
        predicted_pose = imu_prediction.pose() # w2imu
        self.velocity = imu_prediction.velocity()
        self.imu_v_output.append(self.velocity)

        self.graph_initials.insert(gtsam.symbol('v', cur_id), self.velocity)
        self.graph_initials.insert(gtsam.symbol('b', cur_id), self.imu_bias) # TODO? What bias??? # https://github.com/borglab/gtsam/blob/4abef9248edc4c49943d8fd8a84c028deb486f4c/python/gtsam/examples/CombinedImuFactorExample.py#L219C17-L221C55 
        
        predicted_pose_homo = np.eye(4)
        predicted_pose_homo[:3,:3] = predicted_pose.rotation().matrix()
        predicted_pose_homo[:3,3] = predicted_pose.translation()

        if not self.config.imu_pgo:
            self.pim.resetIntegration() # -- preintegration testing --- 
        return predicted_pose_homo
        

    def add_combined_IMU_factor(self,cur_id: int, last_id: int): 
        #
        self.graph_factors.add(gtsam.CombinedImuFactor(gtsam.symbol('x', last_id), gtsam.symbol('v', last_id), gtsam.symbol('x', cur_id),
                                gtsam.symbol('v', cur_id), gtsam.symbol('b', last_id), gtsam.symbol('b', cur_id), self.pim))

    def imu_calibration(self, dataset: SLAMDataset, imu_calibration_steps=30, visual=False, gravity_align=True):
        #
        num_samples = 0
        gyro_avg = np.zeros(3)
        accel_avg = np.zeros(3)

        ang_vel_list = []
        lin_acc_list = []
        timestamps = []

        # for idx in range(int(10*imu_calibration_time)):
        #     if dataset.ts_rawimu[idx].timestamp() - dataset.ts_rawimu[0].timestamp() < imu_calibration_time:
        
        for idx in range(imu_calibration_steps):
            num_samples += 1

            raw_imu_filename = os.path.join(self.config.raw_imu_path, 'data', f'{idx:010d}.txt') #w-17:20, a-11:14
            raw_imu = np.loadtxt(raw_imu_filename)
            lin_acc = raw_imu[11:14]
            ang_vel = raw_imu[17:20]

            gyro_avg += ang_vel
            accel_avg += lin_acc

            ang_vel_list.append(ang_vel)
            lin_acc_list.append(lin_acc)
            timestamps.append(dataset.ts_rawimu[idx].timestamp())

        gyro_avg /= num_samples
        accel_avg /= num_samples
        grav_vec = np.array([0, 0, self.GRAVITY])
        
        if gravity_align:
            # Calculate initial orientation from gravity
            grav_dir = accel_avg / np.linalg.norm(accel_avg) # (normalize to avoid scale, only rot needed)
            grav_target = np.array([0, 0, 1])  # Z-up coordinate system
            initial_orientation, _ = R.align_vectors([grav_target],[grav_dir]) # ?-? b to a, calibrate imu to gravity aligned position
            
            # calibrated_initial_pose = np.eye(4)
            # calibrated_initial_pose[:3, :3] = initial_orientation.as_matrix()#.T
            
            # # rotation seems wrong
            # R_z_90 = np.array([
            #             [0, 1, 0, 0],
            #             [-1, 0, 0, 0],
            #             [0, 0, 1, 0],
            #             [0, 0, 0, 1]
            #         ])
            # calibrated_initial_pose = R_z_90 @ calibrated_initial_pose

            # method 2
            # Calculate angles - Z-axis is up (z = 1 in gravity vector)
            pitch = np.arcsin(-grav_dir[0])  # rotation around y-axis
            roll = np.arcsin(grav_dir[1] / np.cos(pitch))  # rotation around x-axis

            # Create rotation matrices for roll and pitch
            roll_matrix = R.from_euler('x', roll).as_matrix()
            pitch_matrix = R.from_euler('y', pitch).as_matrix()
            
            # Combine roll and pitch into a single rotation matrix
            rotation_matrix = pitch_matrix @ roll_matrix

            # Apply rotation matrix to the calibrated initial pose
            calibrated_initial_pose = np.eye(4)
            calibrated_initial_pose[:3, :3] = rotation_matrix
            #
            print('-----------calibration imu initial pose ---------------',calibrated_initial_pose)

            # Compute biases adjusted by initial pose
            # # grav_corrected = np.dot(calibrated_initial_pose[:3, :3].T, np.array([0, 0, self.GRAVITY]))
            grav_corrected =  np.linalg.inv(calibrated_initial_pose[:3, :3]) @ np.array([0, 0, self.GRAVITY])
            # grav_corrected = initial_orientation.apply(np.array([0, 0, self.GRAVITY])) 
            # grav_corrected = grav_dir * self.GRAVITY
            accel_bias = accel_avg - grav_corrected
            gyro_bias = gyro_avg
        else:
            # # Bias computation using average
            gyro_bias = gyro_avg
            accel_bias = accel_avg - grav_vec
            calibrated_initial_pose = np.eye(4)

        ang_vel_array = np.array(ang_vel_list)
        lin_acc_array = np.array(lin_acc_list)
        timestamps = np.array(timestamps)
        # Calculate max-min range for sigma initialization
        gyro_sigma = (ang_vel_array.max(axis=0) - ang_vel_array.min(axis=0)) / 2
        accel_sigma = (lin_acc_array.max(axis=0) - lin_acc_array.min(axis=0)) / 2
        
        # V- Plotting the IMU data
        if visual:            
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            # Gyroscope data
            axs[0].plot(timestamps, ang_vel_array[:, 0], label='Gyro X', color='r')
            axs[0].plot(timestamps, ang_vel_array[:, 1], label='Gyro Y', color='g')
            axs[0].plot(timestamps, ang_vel_array[:, 2], label='Gyro Z', color='b')
            axs[0].axhline(gyro_bias[0], color='r', linestyle='--', label='Bias X')
            axs[0].axhline(gyro_bias[1], color='g', linestyle='--', label='Bias Y')
            axs[0].axhline(gyro_bias[2], color='b', linestyle='--', label='Bias Z')
            axs[0].set_title('Gyroscope Data')
            axs[0].set_xlabel('Time [s]')
            axs[0].set_ylabel('Angular Velocity [rad/s]')
            axs[0].legend()
            axs[0].grid(True)
            # Display gyro sigmas on the plot
            axs[0].text(0.05, 0.95, f'Gyro Sigma X: {gyro_sigma[0]:.6f}\nGyro Sigma Y: {gyro_sigma[1]:.6f}\nGyro Sigma Z: {gyro_sigma[2]:.6f}',
                    transform=axs[0].transAxes, fontsize=10, verticalalignment='top')
            # Accelerometer data
            axs[1].plot(timestamps, lin_acc_array[:, 0], label='Accel X', color='r')
            axs[1].plot(timestamps, lin_acc_array[:, 1], label='Accel Y', color='g')
            axs[1].plot(timestamps, lin_acc_array[:, 2], label='Accel Z', color='b')
            axs[1].axhline(accel_bias[0], color='r', linestyle='--', label='Bias X')
            axs[1].axhline(accel_bias[1], color='g', linestyle='--', label='Bias Y')
            axs[1].axhline(accel_bias[2] + self.GRAVITY, color='b', linestyle='--', label='Bias Z + Gravity')
            axs[1].set_title('Accelerometer Data')
            axs[1].set_xlabel('Time [s]')
            axs[1].set_ylabel('Linear Acceleration [m/s^2]')
            axs[1].legend()
            axs[1].grid(True)
            # Display accel sigmas on the plot
            axs[1].text(0.05, 0.95, f'Accel Sigma X: {accel_sigma[0]:.6f}\nAccel Sigma Y: {accel_sigma[1]:.6f}\nAccel Sigma Z: {accel_sigma[2]:.6f}',
                    transform=axs[1].transAxes, fontsize=10, verticalalignment='top')

            plt.tight_layout()
            plt.show()

        return calibrated_initial_pose, gyro_bias, accel_bias, accel_sigma, gyro_sigma # array(3)
