# TAP3D Demo


# Shortcuts: 

data collection:
`python data_collection.py --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --collection_duration 1200 --thermal_input m08  --save 1 --inference -1 --use_old_plot 0 --save_dest data/CB328_7`    
or:
`python data_collection.py --exp_config_file model3_m16 --weights weights/m16/model3_m16_thermo_pt_0820152400.pth --collection_duration 1200 --thermal_input m16  --save 1 --inference 1 --use_old_plot 0 --save_dest data/CB328_7`

data check:
`python data_check.py --exp_config_file model3_m16 --weights weights/m16/model3_m16_thermo_pt_0820152400.pth --thermal_input m16 --use_old_plot 1 --no_id_distinguish 0 --img2vid 0 --vis_mode 1 --data_processed 0 --path data/zx_home_3`
or 
`python data_check.py --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --thermal_input m08 --use_old_plot 1 --no_id_distinguish 0 --img2vid 0 --vis_mode 1 --data_processed 0 --path data/zx_home_1`

realtime model:
`python data_collection_single.py --demo-config config/m16.yaml`
or:
`python data_collection_single.py --demo-config config/m08.yaml`
The config file content
1. collection_duration: how long you want to collect data
2. sleep_time: how long between two frames' collection
3. enable_MLX: deprecated
4. mi08/mi16_processing: if we process the data
5. save: if we save the data
6. save_dest: destination for saving data. THe data will be saved in subfolders of this specified folder
7. exp_config_file: config file for model
8. weight: the weight you want to use
9. train
10. thermal_input: which type of sensor as thermal input
11. inference: if we want to predict the 3d points or we just want to collect data
12. use_old_plot: if use old matplotlib plot or new (but slower) plot

14. visualize: if we want to visualize sensor input and results
15. use_recorded_data: if we want to inference from recorded data. If yes, save is automatically false.
16. recorded_data_path: the folder containing recorded data

inferencing from recorded data:
1. keep all yaml file contentes the same
2. change use_recorded_data to 1
3. change recorded_data_path the the folder containing recorded data

annotate and predict (without visualizing point clouds)
`python data_check.py --path data/CB328_7 --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --thermal_input m08 --use_old_plot 0 --no_id_distinguish 0 --img2vid 1 --vis_mode -1 --data_processed 0 --annotate_and_pred 1`

-------------------


# Running demo of TAP3D

- step 0. move model to `weights/m08`

- step 1. plug in m08 and realsense camera

- step 2. run the demo program: `python data_collection.py --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --collection_duration 1200 --thermal_input m08  --save 0 --inference 0 --use_old_plot 1 --save_dest data/meetingRoom0`. 

    - --exp_config_file: the config file for the model, under directory `exp_configs/`

    - --weights: the path to the model weights, under `weights/m08/` directory

    - --collection_duration: the duration of data collection in seconds, default is 60s (1min)

    - --thermal_input: the thermal input source, options are `m08`, `m16` or `seek`

    - --save: whether to save the collected data, 1 for yes, 0 for no

    - --inference: inference mode. 1 is inference+annotate, 0 is annotate only, -1 is none.

    - --use_old_plot: whether to use the old plotting method, 1 for yes, 0 for no.

# Visualize data
## raw data w/o annotation and prediction, visualize raw data + annotated point cloud gt + prediction:
1. assumption: we want to plot both annotation and ground truth point cloud
2. we do not have data annotated or predicted. We only have raw data.
`python data_check.py --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --thermal_input m08 --use_old_plot 1 --no_id_distinguish 0 --img2vid 0 --vis_mode 1 --data_processed 0 --path /home/zx/Desktop/zx/TAP3D_demo/data/CB328_0`
- --path: the folder path
- --exp_config_file: configuration file name
- --weights: path to model weight
- --thermal_input: thermal input type (m08, m16, seek)
- --use_old_plot: whether to use the old plotting method, 1 for yes, 0 for no
- --no_id_distinguish: whether to distinguish between different human, or assign all human an id of 1, 1 for assigning all with 1, 0 for keeping individual, distinct assignment
- --img2vid: whether to convert images to video, 1 for yes, 0 for no
- --vis_mode: display mode. 1 is to show inference+annotated point cloud, -1 is not show.
- --data_processed: whether the data has been annotated and predicted, 1 for yes, 0 for no.

`python data_check.py --exp_config_file model3_m16 --weights weights/m16/model3_m16_thermo_pt_0820152400.pth --thermal_input m16 --use_old_plot 1 --no_id_distinguish 0 --img2vid 0 --vis_mode 1 --data_processed 0 --path /home/zx/Desktop/zx/TAP3D_demo/data/CB328_0`

## raw data w/o annotation and prediction, visualize raw data only: 
`python data_check.py --path /home/zx/Desktop/zx/TAP3D_demo/data/test1 --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --thermal_input m08 --use_old_plot 1 --no_id_distinguish 0 --img2vid 0 --vis_mode -1 --data_processed 0`

`python data_check.py --exp_config_file model3_seekthermal --weights weights/seekthermal/model3_seekthermal_thermo_pt_0820225255.pth --thermal_input seek --use_old_plot 1 --no_id_distinguish 0 --img2vid 0 --vis_mode 1 --data_processed 0 --path /home/zx/Desktop/zx/TAP3D_demo/data/CB328C_0`

## visualize data with annotation and prediction, vidualizing raw data + annotated point cloud gt + prediction:
`python data_check.py --path /home/zx/Desktop/zx/TAP3D_demo/data/HKU --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --thermal_input m08 --use_old_plot 1 --no_id_distinguish 0 --img2vid 0 --vis_mode 1 --data_processed 1`

## check data:
`python data_check.py --path /home/zx/Desktop/zx/TAP3D_demo/data/test_multi --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --thermal_input m08 --use_old_plot 0 --no_id_distinguish 0 --img2vid 0 --vis_mode -1 --data_processed 0`

# annotate and predict
`python data_check.py --path /home/zx/Desktop/zx/TAP3D_demo/data/20260302104102536686 --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --thermal_input m08 --use_old_plot 0 --no_id_distinguish 0 --img2vid 1 --vis_mode -1 --data_processed 0 --annotate_and_pred 1`

## collect data with timestamp as dir name under data/:
`python data_collection.py --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --collection_duration 1200 --save 1`

# A temporary patch:
`python data_collection_single.py --exp_config_file model3_m16 --weights weights/m16/model3_m16_thermo_pt_0820152400.pth --collection_duration 1200 --thermal_input m16  --save 0 --inference 1 --use_old_plot 1`

python data_collection_single.py --exp_config_file model3_m16 --weights weights/m16/model3_m16_thermo_pt_0820152400.pth --collection_duration 1200 --thermal_input m16  --save 0 --inference 1 --use_old_plot 0

TODO
2. gt pcd compare
3. point cloud gt
4. gt for data check


