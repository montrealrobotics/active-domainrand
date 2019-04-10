import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt


npa = np.array

# f = h5py.File('../results/real-robot/real-190328-170248.hdf5', 'r') # smaller demo file with only 2 rollouts per model
f = h5py.File('../results/real-robot/real-190328-174502.hdf5', 'r') # 25 rollouts per model

experiments = ['adr', 'baseline', 'usdr']

for e in experiments:
    print(e + "/distances\t\t", f.get(e + "/distances").shape)
    print(e + "/rewards\t\t", f.get(e + "/rewards").shape)
    print(e + "/images\t\t\t", f.get(e + "/images").shape)
    print(e + "/trajectories\t", f.get(e + "/trajectories").shape)

# example replay:
model_type = "adr"
model_no = 3
run = 1

# for frame in f.get("{}/images".format(model_type))[model_no, run]:
#     if np.count_nonzero(frame) > 0:
#         cv2.imshow("Replay", frame)
#         cv2.waitKey(20)
#
# frame_len = 0
# for frame in f.get("{}/trajectories".format(model_type))[model_no, run]:
#     if np.count_nonzero(frame) > 0:
#         print (np.around(frame,1))
#         frame_len+=1
#
# x = np.arange(frame_len)
#
# for motor in range(4):
#     plt.plot(x, f.get("{}/trajectories".format(model_type))[model_no, run, :frame_len, motor+14], label="motor "+str(motor+1))
#
# plt.plot(x, 5*f.get("{}/distances".format(model_type))[model_no, run, :frame_len], label="distance to goal x 5")
# plt.hlines(0.025*5, 0, frame_len, label="solved", linestyles="dotted")
# plt.ylim((-1,1))
# plt.legend()
# plt.tight_layout()
# plt.show()

# max_frame_len = 0

# for color, model_type in zip(["red", "green", "blue"], experiments):
#     print (model_type, color)
#
#     for model_no in range(5):
#         for run in range(len(f.get("{}/trajectories".format(model_type))[model_no, :])):
#             frame_len = np.count_nonzero(
#                 np.count_nonzero(f.get("{}/trajectories".format(model_type))[model_no, run], axis=1))
#             if frame_len > max_frame_len:
#                 max_frame_len = frame_len
#             x = np.arange(frame_len)
#             plt.plot(x, f.get("{}/distances".format(model_type))[model_no, run, :frame_len], c=color)
#
# plt.hlines(0.025, 0, max_frame_len, label="solved", linestyles="dotted")
# plt.legend()
# plt.tight_layout()
# plt.title("Distances Of All Rollouts Over Time")
# plt.show()




#### HISTOGRAM BAD

# for color, model_type in zip(["red", "green", "blue"], experiments):
#     print (model_type, color)
#     values = []
#
#     for model_no in range(5):
#         for run in range(len(f.get("{}/trajectories".format(model_type))[model_no, :])):
#             frame_len = np.count_nonzero(
#                 np.count_nonzero(f.get("{}/trajectories".format(model_type))[model_no, run], axis=1))
#             values.append(frame_len)
#     plt.hist(values, alpha=0.5, color=color, label=model_type)
#
# plt.legend()
# plt.tight_layout()
# plt.title("Distances Of All Rollouts Over Time")
# plt.show()

#### FINAL DISTANCE PLOT


pos = 1
val = []

colors = ["red", "green", "blue"]

for color, model_type in zip(colors, experiments):
    print (model_type, color)

    values_model = []

    for model_no in range(5):
        for run in range(len(f.get("{}/trajectories".format(model_type))[model_no, :])):
            frame_len = np.count_nonzero(
                np.count_nonzero(f.get("{}/trajectories".format(model_type))[model_no, run], axis=1))
            values_model.append(f.get("{}/distances".format(model_type))[model_no, run, frame_len-1])

    # plt.scatter(np.ones(len(values))*pos, values, alpha=0.5, c=color, label=model_type)

    val.append(values_model)

    pos += 1

bplot = plt.boxplot(npa(val).T, labels=experiments, patch_artist=True)

cm = plt.cm.get_cmap('viridis')
colors = [cm(val/3) for val in range(3)]

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.legend()
# plt.tight_layout()
plt.title("Real Robot Rollout Performance Box Plots\n"
          "5 policies per approach, 25 runs per policy")
plt.show()


