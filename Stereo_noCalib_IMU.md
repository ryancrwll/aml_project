# MODEL USED: stereo ebc WITHOUT calib file and using sensor fusion with imu data

# TRAINING LOG:

Total training sequences: 6208
Checkpoints will be saved to: ./checkpoints
Epoch 1/30: 100%|████████████████████████| 1552/1552 [03:14<00:00,  8.00it/s, event=3.56e-8, imu=0.00117, loss=0.00276]
Epoch 1 Average Loss: 0.00275972
Epoch 2/30: 100%|███████████████████████| 1552/1552 [03:17<00:00,  7.84it/s, event=5.11e-8, imu=0.000755, loss=0.00152]
Epoch 2 Average Loss: 0.00152472
Epoch 3/30: 100%|██████████████████████| 1552/1552 [03:15<00:00,  7.93it/s, event=1.73e-8, imu=0.000448, loss=0.000981]
Epoch 3 Average Loss: 0.00098089
Epoch 4/30: 100%|████████████████████████| 1552/1552 [03:17<00:00,  7.86it/s, event=3.2e-9, imu=0.000349, loss=0.00072]
Epoch 4 Average Loss: 0.00071918
Epoch 5/30: 100%|██████████████████████| 1552/1552 [03:16<00:00,  7.89it/s, event=3.91e-9, imu=0.000887, loss=0.000622]
Epoch 5 Average Loss: 0.00062210
SAVED checkpoint: ./checkpoints/deio_model_ep5.pth
Epoch 6/30: 100%|██████████████████████| 1552/1552 [03:15<00:00,  7.95it/s, event=8.78e-9, imu=0.000319, loss=0.000538]
Epoch 6 Average Loss: 0.00053816
Epoch 7/30: 100%|█████████████████████| 1552/1552 [03:19<00:00,  7.78it/s, event=5.18e-10, imu=0.000473, loss=0.000454]
Epoch 7 Average Loss: 0.00045410
Epoch 8/30: 100%|███████████████████████| 1552/1552 [03:19<00:00,  7.77it/s, event=1.36e-8, imu=0.00014, loss=0.000404]
Epoch 8 Average Loss: 0.00040356
Epoch 9/30: 100%|██████████████████████| 1552/1552 [03:18<00:00,  7.84it/s, event=5.77e-9, imu=0.000308, loss=0.000369]
Epoch 9 Average Loss: 0.00036830
Epoch 10/30: 100%|██████████████████████| 1552/1552 [03:16<00:00,  7.88it/s, event=2.62e-8, imu=0.00023, loss=0.000319]
Epoch 10 Average Loss: 0.00031919
SAVED checkpoint: ./checkpoints/deio_model_ep10.pth
Epoch 11/30: 100%|██████████████████████| 1552/1552 [03:15<00:00,  7.93it/s, event=9.17e-8, imu=0.000367, loss=0.00028]
Epoch 11 Average Loss: 0.00028017
Epoch 12/30: 100%|██████████████████████| 1552/1552 [03:16<00:00,  7.89it/s, event=1.41e-9, imu=0.00018, loss=0.000267]
Epoch 12 Average Loss: 0.00026654
Epoch 13/30: 100%|█████████████████████| 1552/1552 [03:18<00:00,  7.82it/s, event=6.72e-9, imu=0.000187, loss=0.000224]
Epoch 13 Average Loss: 0.00022402
Epoch 14/30: 100%|██████████████████████| 1552/1552 [03:18<00:00,  7.82it/s, event=4.37e-8, imu=9.39e-5, loss=0.000218]
Epoch 14 Average Loss: 0.00021808
Epoch 15/30: 100%|██████████████████████| 1552/1552 [03:18<00:00,  7.81it/s, event=1.55e-9, imu=8.71e-5, loss=0.000198]
Epoch 15 Average Loss: 0.00019836
SAVED checkpoint: ./checkpoints/deio_model_ep15.pth
Epoch 16/30: 100%|█████████████████████| 1552/1552 [03:17<00:00,  7.84it/s, event=1.69e-9, imu=0.000141, loss=0.000192]
Epoch 16 Average Loss: 0.00019232
Epoch 17/30: 100%|█████████████████████| 1552/1552 [03:16<00:00,  7.89it/s, event=9.71e-10, imu=7.85e-5, loss=0.000176]
Epoch 17 Average Loss: 0.00017557
Epoch 18/30: 100%|███████████████████████| 1552/1552 [03:19<00:00,  7.78it/s, event=3.22e-8, imu=9.3e-5, loss=0.000159]
Epoch 18 Average Loss: 0.00015889
Epoch 19/30: 100%|█████████████████████| 1552/1552 [03:18<00:00,  7.83it/s, event=3.61e-8, imu=0.000171, loss=0.000148]
Epoch 19 Average Loss: 0.00014819
Epoch 20/30: 100%|█████████████████████| 1552/1552 [03:20<00:00,  7.75it/s, event=3.16e-9, imu=0.000105, loss=0.000151]
Epoch 20 Average Loss: 0.00015127
SAVED checkpoint: ./checkpoints/deio_model_ep20.pth
Epoch 21/30: 100%|█████████████████████| 1552/1552 [03:16<00:00,  7.90it/s, event=6.22e-9, imu=0.000125, loss=0.000146]
Epoch 21 Average Loss: 0.00014547
Epoch 22/30: 100%|█████████████████████| 1552/1552 [03:17<00:00,  7.86it/s, event=9.95e-10, imu=5.96e-5, loss=0.000167]
Epoch 22 Average Loss: 0.00016718
Epoch 23/30: 100%|████████████████████| 1552/1552 [03:18<00:00,  7.84it/s, event=7.95e-10, imu=0.000118, loss=0.000122]
Epoch 23 Average Loss: 0.00012179
Epoch 24/30: 100%|██████████████████████| 1552/1552 [03:18<00:00,  7.80it/s, event=2.27e-8, imu=4.77e-5, loss=0.000121]
Epoch 24 Average Loss: 0.00012099
Epoch 25/30: 100%|██████████████████████| 1552/1552 [03:18<00:00,  7.82it/s, event=2.27e-9, imu=7.04e-5, loss=0.000116]
Epoch 25 Average Loss: 0.00011584
SAVED checkpoint: ./checkpoints/deio_model_ep25.pth
Epoch 26/30: 100%|███████████████████████| 1552/1552 [03:20<00:00,  7.74it/s, event=1.45e-8, imu=6.4e-5, loss=0.000112]
Epoch 26 Average Loss: 0.00011242
Epoch 27/30: 100%|█████████████████████| 1552/1552 [03:18<00:00,  7.83it/s, event=6.12e-10, imu=4.57e-5, loss=0.000131]
Epoch 27 Average Loss: 0.00013078
Epoch 28/30: 100%|███████████████████████| 1552/1552 [03:18<00:00,  7.82it/s, event=1.33e-8, imu=5.48e-5, loss=9.96e-5]
Epoch 28 Average Loss: 0.00009954
Epoch 29/30: 100%|██████████████████████| 1552/1552 [03:18<00:00,  7.80it/s, event=1.29e-8, imu=0.00013, loss=0.000104]
Epoch 29 Average Loss: 0.00010442
Epoch 30/30: 100%|██████████████████████| 1552/1552 [03:16<00:00,  7.89it/s, event=2.55e-10, imu=6.2e-5, loss=0.000109]
Epoch 30 Average Loss: 0.00010949
SAVED checkpoint: ./checkpoints/deio_model_ep30.pth

# EVALUATION:

--- Starting Evaluation ---
Mode: STEREO | UNCALIBRATED
IMU Input: ENABLED
DEIO Model loaded from ./checkpoints/deio_model_ep30.pth. Channels: 10
Running inference and state estimation...
100%|██████████████████████████████████████████████████████████████████████████████| 2196/2196 [01:58<00:00, 18.56it/s]
Aligning trajectories...

--- DEIO Evaluation Metrics ---
Total Trajectory Steps: 21960 steps
(21960, 3)
Sim(3) Scale Factor: 1.0016
Trajectory RMSE (ALIGNED): 0.0709 meters

Generating plot...
GT shape: (21960, 3), min: [-1.6873883 -3.4663277 -0.8056722], max: [0.5656026  0.02039042 3.62572   ]
Pred aligned shape: (21960, 3), min: [-1.7169633 -3.4794405 -0.8714336], max: [0.57478803 0.05018038 3.6810143 ]
Plot saved to deio_trajectory_comparison_aligned.png

![alt text](results_Stereo_noCalib_IMU.png)