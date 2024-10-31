# # Let's load the CSV file to examine its content and perform the required operation.
# import pandas as pd

# # Load the file to inspect its content
# file_path = 'data/Newer_College_Dataset/mine_easy/medium_gt_state_tum_corrected.csv'
# with open(file_path, 'r') as file:
#     file_content = file.readlines()

# # Process each line to replace the first space with a period
# modified_content = [line.replace(' ', '.', 1) for line in file_content]

# # Write the modified content to a new CSV file
# output_path = 'data/Newer_College_Dataset/mine_easy/medium_gt_state_tum_corrected.csv'
# with open(output_path, 'w') as file:
#     file.writelines(modified_content)

# output_path


from evo.core import trajectory, metrics, sync
from evo.tools import plot, file_interface

import copy
import matplotlib.pyplot as plt

# -----------------nc_mine dataset -------------------
traj_est_file_origin = 'experiments/history/pretrained/ncm-kitti360 4-5s/nc_mine-test_ros_2024-10-31_08-40-08/slam_poses__tum.txt'
traj_ref_file = 'data/Newer_College_Dataset/mine_easy/medium_gt_state_tum_corrected.csv'
# ------------------ evaluate -------------------
traj_est_file = traj_est_file_origin + '_transformed2GT'

traj_est = file_interface.read_tum_trajectory_file(traj_est_file) # -> PoseTrajectory3D
traj_ref = file_interface.read_tum_trajectory_file(traj_ref_file)

print(traj_est.num_poses)
# traj_ref_downsample = copy.deepcopy(traj_ref)
# # traj_ref_downsample.downsample(traj_est.num_poses)
traj_ref_downsample, traj_est_aligned_scaled = sync.associate_trajectories(traj_ref, traj_est)
# traj_est_aligned_scaled = copy.deepcopy(traj_est)
traj_est_aligned_scaled.align(traj_ref_downsample, correct_scale=False) # do not align scale
# # traj_est_aligned_scaled = trajectory.align_trajectory(traj_est_aligned_scaled, traj_ref_downsample, correct_scale=False)


fig = plt.figure(figsize=(8, 16))
plot_mode = plot.PlotMode.xy

ax = plot.prepare_axis(fig, plot_mode)
plot.traj(ax, plot_mode, traj_ref_downsample, '--', 'gray')
plot.traj(ax, plot_mode, traj_est_aligned_scaled, '-', 'blue')
plot.traj(ax, plot_mode, traj_est, '-', 'red') # original
fig.axes.append(ax)
plt.title('$\mathrm{SE}(3)$ alignment')

plt.show()

# # --- Compute the ATE
data = (traj_ref_downsample, traj_est_aligned_scaled)
ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
ape_metric.process_data(data)
ape_statistics = ape_metric.get_all_statistics()
print("mean:", ape_statistics["mean"])
print(ape_statistics)

# -- visualize error 

print("plotting")
plot_collection = plot.PlotCollection("Example")

# metric values
fig_1 = plt.figure(figsize=(8, 8))
plot_mode = plot.PlotMode.xy
ax = plot.prepare_axis(fig_1, plot_mode)
plot.error_array(ax, ape_metric.error, statistics=ape_statistics,
                 name="APE", title=str(ape_metric))
plot_collection.add_figure("raw", fig_1)

# trajectory colormapped with error
fig_2 = plt.figure(figsize=(8, 8))
plot_mode = plot.PlotMode.xy
ax = plot.prepare_axis(fig_2, plot_mode)
plot.traj(ax, plot_mode, traj_ref_downsample, '--', 'gray', 'reference')
plot.traj_colormap(
    ax, traj_est_aligned_scaled, ape_metric.error, plot_mode, min_map=ape_statistics["min"],
    max_map=ape_statistics["max"], title="APE mapped onto trajectory")
plot_collection.add_figure("traj (error)", fig_2)

# trajectory colormapped with speed
fig_3 = plt.figure(figsize=(8, 8))
plot_mode = plot.PlotMode.xy
ax = plot.prepare_axis(fig_3, plot_mode)
speeds = [
    trajectory.calc_speed(traj_est_aligned_scaled.positions_xyz[i],
                          traj_est_aligned_scaled.positions_xyz[i + 1],
                          traj_est_aligned_scaled.timestamps[i], traj_est_aligned_scaled.timestamps[i + 1])
    for i in range(len(traj_est_aligned_scaled.positions_xyz) - 1)
]
speeds.append(0)
plot.traj(ax, plot_mode, traj_ref_downsample, '--', 'gray', 'reference')
plot.traj_colormap(ax, traj_est_aligned_scaled, speeds, plot_mode, min_map=min(speeds),
                   max_map=max(speeds), title="speed mapped onto trajectory")
fig_3.axes.append(ax)
plot_collection.add_figure("traj (speed)", fig_3)

plot_collection.show()

