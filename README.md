# Deep Event Inertial Odometry (DEIO)

This repository implements a Deep Event Inertial Odometry system using PyTorch. It processes event camera streams (from MVSEC) and IMU data to estimate 6-DOF pose, comparing performance against frame-based stereo baselines.

## Prerequisites

* **Python 3.7+**
* **PyTorch** (tested on 1.10+)
* **NumPy, SciPy, h5py, OpenCV**
* **Matplotlib, Tqdm**

Install dependencies:
```bash
pip install torch numpy scipy h5py opencv-python matplotlib tqdm pyyaml

Download the MVSEC dataset (Indoor Flying sequences).

Place the .hdf5 files in a ./data folder in the root directory.

Ensure you have matching pairs: *_data.hdf5 and *_gt.hdf5

Training (deio_train.py)

    Trains the network on sequences defined in TRAIN_DATASETS.

    Input: Voxel grids (events) + IMU (optional).

    Output: Saves model weights to ./checkpoints.

Evaluation (deio_eval.py)

    Evaluates a specific checkpoint on a test sequence.

    Calculates RMSE (Root Mean Square Error) after Sim(3) alignment.

    Generates a trajectory plot deio_trajectory_aligned.png.

Comparison (compare.py)

    Runs a head-to-head comparison between:

    DEIO (Your Model): Event-based estimation.

    Frame VO (Baseline): Standard feature-tracking stereo visual odometry using actual image frames from the dataset.