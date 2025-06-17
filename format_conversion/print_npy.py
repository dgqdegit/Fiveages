import numpy as np
path = "/home/zk/Projects/camera_calibration/calibration_results_left/external_matrices/extrinsic_matrix.npy"
data = np.load(path)
print(data)
print(data.shape)
print(type(data))
