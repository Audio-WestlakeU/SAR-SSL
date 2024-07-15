## Quick start
### Pretext Task
+ **Data generation**
  - Simulated data
    ```
    python data_generation_SimulatedSIG_notspecifyroom.py --stage pretrain --wnoise --gpu-id [*]
    python data_generation_SimulatedSIG_notspecifyroom.py --stage preval --wnoise --gpu-id [*] 
    python data_generation_SimulatedSIG_notspecifyroom.py --stage test --wnoise --gpu-id [*]
    ```
  - Real-world data (DCASE, MeshRIR, MIR, ACE, dEchorate, BUTReverb)
    1. select recorded RIRs and noise signals
    ```
    python data_generation_MeasuredRIR.py --data-id 0 --data-type rir noise # DCASE
    python data_generation_MeasuredRIR.py --data-id 3 --data-type rir noise # ACE
    python data_generation_MeasuredRIR.py --data-id 4 --data-type rir noise # dEchorate
    python data_generation_MeasuredRIR.py --data-id 5 --data-type rir noise # BUTReverb
    python data_generation_MeasuredRIR.py --data-id 1 --data-type rir # MeshRIR 
    python data_generation_MeasuredRIR.py --data-id 2 --data-type rir # MIR
    ```
    2. generate microphone signals with recorded RIRs and noise signals 
    ```
    python data_generation_SIGfromMeasuredRIR.py --data-id 0 3 4 5 --wnoise --stage pretrain 
    python data_generation_SIGfromMeasuredRIR.py --data-id 0 3 4 5 --wnoise --stage preval
    python data_generation_SIGfromMeasuredRIR.py --data-id 0 3 4 5 --wnoise --stage test
    python data_generation_SIGfromMeasuredRIR.py --data-id 1 2 --stage pretrain 
    python data_generation_SIGfromMeasuredRIR.py --data-id 1 2 --stage preval
    python data_generation_SIGfromMeasuredRIR.py --data-id 1 2 --stage test
    ```
  - Real-world data (LOCATA)
    ```
    python data_generation_LOCATA.py --stage pretrain
    python data_generation_LOCATA.py --stage preval
    python data_generation_LOCATA.py --stage test_pretrain
    ```
  - Simulated data (some instances)
    1. uncomment `acoustic_scene.dp_mic_signal = []` in class `RandomMicSigDatasetOri` of `data_generation_dataset.py`
    2. specify `room_size`, `T60`, `SNR` in `data_generation_opt.py` (default)
    3. generate corresponding intances
    ```
    python data_generation_SimulatedSIG_notspecifyroom.py --stage test --wnoise --ins --gpu-id 7 
    ```

+ **Training**
  
  Sepcify the data time version (`self.time_ver`) and whether training with simulated data (`self.pretrain_sim`) in class `opt_pretrain` of `opt.py`. 
  When using real-world data, first train on simulated data with a default cosine-decay learing rate (initialized with 0.001), and then finetune on real-world data with a learning rate 0.0001.
  
  ```
  python run_pretrain.py --pretrain --gpu-id [*]
  ```
+ **Evaluation**

  Specify test_mode in run_pretrain.py
  ```
  python run_pretrain.py --test --time [*] --gpu-id [*]
  ```
+ **Trained models**
  - best_model.tar

### Downstream Task
+ **Data generation**
  - Simulated data
    1. generate RIRs 
    ```
    python data_generation_SimulatedRIR.py --gpu-id [*]
    ```
    2. generate microphone signals from RIRs
    ```
    # room = 2, 4, 8, 16, 32, 64, 128 or 256, and room-trial-id = 16, 8, 4, 2, 1, 1, 1 or 1
    python data_generation_SIGfromMeasuredRIR.py --data-id 6 --wnoise --stage train --room 8 --room-trial-id 0 
    python data_generation_SIGfromMeasuredRIR.py --data-id 6 --wnoise --stage val --room 20 
    python data_generation_SIGfromMeasuredRIR.py --data-id 6 --wnoise --stage test --room 20 
    ```
    | Stage | Trials   | nRooms | nRIRs/Room | nSrcSig/RIR | nMicSig |
    |:----- |:-------- |:------ |:---------- |:----------- |:------- |
    | train | x16      | 2      | 50         | 2           | 200     |
    |       | x8       | 4      | 50         | 2           | 400     |
    |       | x4       | 8      | 50         | 2           | 800     |
    |       | x2       | 16     | 50         | 2           | 1600    |
    |       | x1       | 32     | 50         | 2           | 3200    |
    |       | x1       | 64     | 50         | 2           | 6400    |
    |       | x1       | 128    | 50         | 2           | 12800   |
    |       | x1       | 256    | 50         | 2           | 25600   |
    | val   | -        | 20     | 50         | 1           | 1000    |
    | test  | -        | 20     | 50         | 4           | 4000    |

  - Real-world data
    - TDOA estimation
    ```
    python data_generation_LOCATA.py --stage train
    python data_generation_LOCATA.py --stage val 
    python data_generation_LOCATA.py --stage test 
    ```
    - DRR, T60, C50, absorption coefficient estimation: on-the-fly from selected RIRs and noise signals

+ **Training**

  Sepcify the data time version (`self.time_ver`) and whether training with simulated data (`downstream_sim`) in class `opt_downstream` of `opt.py`
  - Simulated data
  ```
  # ds-nsimroom = 2, 4, 8, 16, 32, 64, 128 or 256
  # ds-trainmode = finetune, lineareval or scratchLOW
  python run_downstream.py --ds-train --ds-trainmode finetune --ds-nsimroom 8 --ds-task TDOA --time [*] --gpu-id [*] 
  python run_downstream.py --ds-train --ds-trainmode finetune --ds-nsimroom 8 --ds-task DRR T60 C50 ABS --time [*] --gpu-id [*] 

  python run_downstream.py --ds-train --ds-trainmode scratchUP --ds-task TDOA --time [*] --gpu-id [*] 
  python run_downstream.py --ds-train --ds-trainmode scratchUP --ds-task DRR T60 C50 ABS --time [*] --gpu-id [*] 
  ```
  - Real-world data
  ```
  # ds-trainmode = finetune or scratchLOW
  # ds-real-sim-ratio = 1 1, 1 0 or 0 1
  python run_downstream.py --ds-train --ds-trainmode finetune --ds-real-sim-ratio 1 1 --ds-task TDOA --time [*] --gpu-id [*]
  python run_downstream.py --ds-train --ds-trainmode finetune --ds-real-sim-ratio 1 1 --ds-task DRR T60 C50 ABS --time [*] --gpu-id [*]

  ```
+ **Evaluation**

  Specify test mode (`test_mode`) in `run_downstream.py`
  - Simulated data
  ```
  # ds-nsimroom = 2, 4, 8, 16, 32, 64, 128 or 256
  # ds-trainmode = finetune, lineareval or scratchLOW
  python run_downstream.py --ds-test --ds-trainmode finetune --ds-nsimroom 8 --ds-task TDOA --time [*] --gpu-id [*] 
  python run_downstream.py --ds-test --ds-trainmode finetune --ds-nsimroom 8 --ds-task DRR T60 C50 ABS --time [*] --gpu-id [*] 

  python run_downstream.py --ds-test --ds-trainmode scratchUP --ds-task TDOA --time [*] --gpu-id [*] 
  python run_downstream.py --ds-test --ds-trainmode scratchUP --ds-task DRR T60 C50 ABS --time [*] --gpu-id [*] 
  ```
  - Real-world data
  ```
  # ds-trainmode = finetune or scratchLOW
  # ds-real-sim-ratio = 1 1, 1 0 or 0 1
  python run_downstream.py --ds-test --ds-trainmode finetune --ds-real-sim-ratio 1 1 --ds-task TDOA --time [*] --gpu-id [*] 
  python run_downstream.py --ds-test --ds-trainmode finetune --ds-real-sim-ratio 1 1 --ds-task DRR T60 C50 ABS --time [*] --gpu-id [*] 
  ```
  - Read downstream results (MAEs of TDOA, DRR, T60, C50, SNR, ABS estimation) from saved mat files
  ```
  python read_dsmat_bslr.py --time [*]
  python read_lossmetric_simdata.py
  python read_lossmetric_realdata.py
  ```

+ **Trained models**
  - ensemble_model.tar
