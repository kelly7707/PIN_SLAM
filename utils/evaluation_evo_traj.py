# evo_traj tum traj_1.txt --ref  --align --correct_scale
from evo.core import trajectory, metrics, sync
from evo.tools import plot, file_interface

import copy
import matplotlib.pyplot as plt

# # ----------------New College Dataset--------------------
# # traj_est_file = 'experiments/history/!new college norm 800warmup 0.2dropout test_ros_2024-08-08_09-07-45/odom_poses__tum.txt'
# traj_est_file = 'experiments/history/tanh-new college-no dropout-test_ros_2024-08-22_22-16-16/slam_poses__tum.txt'
# traj_est_file = 'experiments/history/corrected-initial_guess-deskewing-pgo-test_ros_2024-06-26_13-37-37/odom_poses__tum.txt'

# traj_est_file = 'experiments/test_ros_2024-09-09_17-35-43/slam_poses__tum.txt' # with relu

# # -- unique model testing
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/ours-newcollege_test_ros_2024-09-07_11-48-24/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/newcollege-pgocorrected-pin-slam_test_ros_2024-09-07_15-54-36/slam_poses__tum_correctedts.txt'
# # traj_est_file = 'experiments/history/tempororily_unique_model/nce-ours-withsmallerlamda-test_ros_2024-09-09_10-15-25/slam_poses__tum.txt'
# # traj_est_file = 'experiments/history/tempororily_unique_model/nce-zeroquery-test_ros_2024-09-11_09-27-00/slam_poses__tum.txt'

# # -- adpted unique model
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/8dim-V-nce-test_ros_2024-09-17_10-22-50/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce-10mapping iter - test_ros_2024-09-12_19-24-37/slam_poses__tum.txt'

# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/9neigh-nce-test_ros_2024-09-13_06-11-52/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce-13 mapping iter-calculated warmup - test_ros_2024-09-18_16-54-17/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce-20mapping iter - test_ros_2024-09-18_11-01-00/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce-5neighbors-test_ros_2024-09-11_18-46-01/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/12 neighbors-nce-test_ros_2024-09-19_19-29-10/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce-freeze in certain frequency-test_ros_2024-09-23_09-10-18/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/test_ros_2024-09-23_11-03-17/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce-3500-3500-3000test_ros_2024-09-23_23-15-24/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce-new_recent_allhis-1500-1500--test_ros_2024-09-24_12-06-40/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce-new_recent_allhis-1500_3000--test_ros_2024-09-24_14-17-56/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce-new samples for training around 03-test_ros_2024-09-25_19-10-28/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce-history surface 06-test_ros_2024-09-25_21-03-50/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/!nce- unfreeze50050- history_idx 06 around surface-test_ros_2024-09-26_09-05-52/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce- sdf_normal_03- test_ros_2024-10-10_08-12-11/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/nce-sdf_normal_03-based_on_baseline-test_ros_2024-10-10_11-12-50/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/test_ros_2024-10-26_23-30-10_ nce_basic pretrained/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/test_ros_2024-10-27_11-08-40_nce_basic pretrain_wo warmup/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/test_ros_2024-10-27_15-32-48/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/pretrained/ncm-kitti360 4-5s/nce-pretrained ncm_kitti360 - no warmup- freeze_unfreeze 50_500-test_ros_2024-10-30_15-48-04/slam_poses__tum.txt'

# traj_ref_file = 'data/Newer_College_Dataset/gt-nc-quad-easy_TMU.csv' # ground truth in tum format
# # traj_ref_file = 'data/Newer_College_Dataset/transformed_gt-nc-quad-easy_TMU.csv'

# # ------------------ new college_medium dataset -------------------
# traj_est_file = 'experiments/history/tempororily_unique_model/newcollege_medium-KQ-tanh_V-relu_test_ros_2024-09-08_17-53-19/slam_poses__tum.txt' # just play around
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/newcollege_medium-ours-test_ros_2024-09-08_21-07-51/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/newcolmedium-pinslam-test_ros_2024-09-08_23-17-28/slam_poses__tum_correctedts.txt'

# traj_est_file = 'experiments/history/tempororily_unique_model/ncm-ours-smalllamda-test_ros_2024-09-09_15-56-52/slam_poses__tum.txt'
# # -- overfitting testing
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/ncm-unfreeze500-50-test_ros_2024-09-23_17-55-34/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/!ncm- unfreeze50050- history_idx 06 around surface-test_ros_2024-09-26_12-14-17/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/pretrained/ncm-kitti360 4-5s/ncm-test_ros_2024-10-30_21-47-25/slam_poses__tum.txt'

# traj_ref_file = 'data/Newer_College_Dataset/medium/gt-nc-quad-medium.csv'
# # traj_ref_file = 'data/Newer_College_Dataset/transformed_gt-nc-quad-medium.csv'


# ------------------ Katzensee Dataset-------------------
# traj_est_file = 'experiments/history/asl_dataset/katzensee pinslam corrected-poses saved test_ros_2024-08-19_10-39-09/slam_poses__tum.txt'
# # traj_est_file = 'experiments/history/asl_dataset/tatzensee-tanh-attention-no dropout-test_ros_2024-08-20_19-05-00/slam_poses__tum.txt'
# traj_est_file = 'experiments/test_ros_2024-08-23_08-58-19/slam_poses__tum.txt'

# - mapper iter 50; eikonal loss downsample 1
# traj_est_file = 'experiments/history/asl_dataset/mapping-iter-50/attention-mapper iter 50/slam_poses__tum.txt' # ours- mapper iter 50; eikonal loss downsample 1
# traj_est_file ='experiments/history/asl_dataset/mapping-iter-50/pin-slam- mapping 50- ekional 1 -test_ros_2024-08-28_17-42-15/slam_poses__tum_correctedts.txt' # pin-slam

# - mesh comparison (pinslam v.s. our stable version) (wrong pgo!!)
# traj_est_file = 'experiments/history/asl_dataset/meshtesting-attention-stable-test_ros_2024-09-01_10-54-45/slam_poses__tum.txt'
# traj_est_file = 'experiments/history/asl_dataset/katzensee pinslam corrected-poses saved test_ros_2024-08-19_10-39-09/slam_poses__tum_correctedts.txt'
# - corrected pgo, mesh compare
# traj_est_file = 'experiments/history/asl_dataset/corrected_pgo-stable-test_ros_2024-09-04_19-13-32/slam_poses__tum.txt' # our stable
# traj_est_file = 'experiments/history/asl_dataset/corrected_saved_pgoposes-drooput02-test_ros_2024-09-04_17-42-00/slam_poses__tum.txt' # our dropout 0.2, stable, but loss high

# - unique testing (larger lambda 1e-3, with dropout and tanh,)
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/katzensee-ours-unique-test_ros_2024-09-07_10-20-20/slam_poses__tum.txt'
# traj_est_file = 'experiments/history/tempororily_unique_model/12geo_feature-ours-test_ros_2024-09-07_21-20-58/slam_poses__tum.txt'

# traj_est_file_origin = 'experiments/history/asl_dataset/corrected_pgo_poses-pinslam-test_ros_2024-09-04_22-58-16/slam_poses__tum_correctedts.txt' # pinslam

# -- adapted unique model
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/12_mapping_iter-katzensee-test_ros_2024-09-12_22-56-08/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/9neighbors-katzensees-test_ros_2024-09-13_00-13-05/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/katzensee-20mapping-calculated warmup-test_ros_2024-09-19_09-57-11/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/katzensee-12neighbors-test_ros_2024-09-19_11-19-22/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/katzensee-4neighbors-test_ros_2024-09-19_12-46-22/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/katzensee-unfreeze in certain frequency500-50 - test_ros_2024-09-23_15-40-55/slam_poses__tum.txt' # 0.479304357137083
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/katzensee-new-recent-1500-1500-allhistory-test_ros_2024-09-24_10-50-18/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/katzensee-history distance to surface 06-test_ros_2024-09-25_22-56-23/slam_poses__tum.txt' # better 0.3553936312913024
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/! katzesee- unfreeze50050- history_idx 06 around surface-test_ros_2024-09-26_10-37-51/slam_poses__tum.txt' # better 0.3883542329867946 
# traj_est_file_origin = 'experiments/history/tempororily_unique_model/testing/katzensee-sdf_normal_03-test_ros_2024-10-09_23-46-12/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/test_ros_2024-10-30_10-19-09/slam_poses__tum.txt'
# traj_est_file_origin = 'experiments/history/pretrained/ncm-kitti360 4-5s/katzensee-no warmup- freeze_unfreeze-test_ros_2024-10-30_14-18-42/slam_poses__tum.txt'

# traj_ref_file = 'data/ASL/katzensee/gt-katzensee_s.csv'
# # traj_ref_file = 'data/ASL/katzensee/transformed_gt-katzensee_s.csv'

# ------------------ nc_math dataset -------------------
# traj_est_file_origin = 'experiments/history/pretrained/ncm-kitti360 4-5s/nc_math - test_ros_2024-10-30_23-42-41/slam_poses__tum.txt' # our pretrain-kitti360 ncm --> no warmup, freeze_unfreeze 50_500
# traj_est_file_origin = 'experiments/history/pretrained/ncm-kitti360 4-5s/pinslam- nc_math-test_ros_2024-10-31_07-53-33/slam_poses__tum_correctedts.txt' # pinslam

# traj_ref_file = 'data/Newer_College_Dataset/math_easy/gt_math_easy.csv'

# -----------------nc_mine dataset -------------------
traj_est_file_origin = 'experiments/history/pretrained/ncm-kitti360 4-5s/nc_mine-test_ros_2024-10-31_08-40-08/slam_poses__tum.txt'
traj_ref_file = 'data/Newer_College_Dataset/mine_easy/medium_gt_state_tum.csv'
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






# # ----------------------------------------
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
# file1_path = 'experiments/history/pretrained/ncm-kitti360 4-5s/pinslam- nc_math-test_ros_2024-10-31_07-53-33/slam_poses__tum_correctedts.txt'  # Path to the pin-slam copy file
# file2_path = 'experiments/history/pretrained/ncm-kitti360 4-5s/nc_math - test_ros_2024-10-30_23-42-41/slam_poses__tum.txt'  # Path to the our file
# output_file_path = 'experiments/history/pretrained/ncm-kitti360 4-5s/pinslam- nc_math-test_ros_2024-10-31_07-53-33/slam_poses__tum_correctedts.txt'  # Path to the output file where the result will be saved

# replace_first_column(file1_path, file2_path, output_file_path)