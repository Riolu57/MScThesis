import numpy as np
import torch

SEED = 123456
EEG_DATA_PATH = "./training/data/eeg/MRCP_data_av_conditions.mat"
KIN_DATA_PATH = "./training/data/kin"
EPOCHS = 5
PRE_TRAIN = 2
LEARNING_RATE = 1e-4
ALPHA = 1
DTYPE_NP = np.float32
DTYPE_TORCH = torch.float32
EEG_CHANNELS = [8, 45, 9, 12, 13, 14, 51, 18, 52, 19, 53, 23, 56, 24, 57, 25]
FLOAT_TOLERANCE = 0.00001
