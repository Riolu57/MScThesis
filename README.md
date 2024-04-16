## Grasping Undercover: EEG Embeddings' Similarities to Kinematics Data

 ### Preparations
1. Make sure you have EEG and Kinematics grasping data from [Sburlea et al. (2021)](https://www.sciencedirect.com/science/article/pii/S2666956021000106).
The expected format is as follows:
```
training
├── data
│   ├── eeg
│   │   └── MRCP_data_av_conditions.mat
│   └── kin
│       ├── subject3
│       │   ├── DataMeasures_LC_S3.mat
│       │   ├── DataMeasures_LS_S3.mat
│       │   ├── DataMeasures_SC_S3.mat
│       │   ├── DataMeasures_SS_S3.mat
│       │   └── indexes_trials.mat
│       ├── subject4
│       │   ├── DataMeasures_LC_S4.mat
│       │   ├── DataMeasures_LS_S4.mat
│       │   ├── DataMeasures_SC_S4.mat
│       │   ├── DataMeasures_SS_S4.mat
│       │   └── indexes_trials.mat
    ...
│       └── subject18
│           ├── DataMeasures_LC_S3.mat
│           ├── DataMeasures_LS_S3.mat
│           ├── DataMeasures_SC_S3.mat
│           ├── DataMeasures_SS_S3.mat
│           └── indexes_trials.mat
├── models
│   └── ...
└── results
    └── ...

```
2. If you want to use an NVIDIA GPU, install the ``requirements_nvidia.txt`` (CUDA 12.1 is expected to have been installed, as well as the general setup of installing NVIDIA tools. A guide can be found further down here: [PyTorch guide](https://pytorch.org/get-started/locally/)).
    Otherwise, or if you are unsure, install the ``requirements.txt``.
3. Change and adjust ``CONFIG.py`` to your liking, as well as uncomment/comment out code in ``main.py`` as you see fit.
4. Run ``main.py``.