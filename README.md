# Running demo of TAP3D

- step 0. move model to `weights/m08`

- step 1. plug in m08 and realsense camera

- step 2. run the demo program: `python data_collection.py --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --collection_duration 1200 --thermal_input m08  --save 0 --inference 0 --use_old_plot 1`. 

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
`python data_check.py --path /home/zx/Desktop/zx/TAP3D_demo/data/test1 --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --thermal_input m08 --use_old_plot 1 --no_id_distinguish 0 --img2vid 0 --vis_mode 1 --data_processed 0`
- --path: the folder path
- --exp_config_file: configuration file name
- --weights: path to model weight
- --thermal_input: thermal input type (m08, m16, seek)
- --use_old_plot: whether to use the old plotting method, 1 for yes, 0 for no
- --no_id_distinguish: whether to distinguish between different human, or assign all human an id of 1, 1 for assigning all with 1, 0 for keeping individual, distinct assignment
- --img2vid: whether to convert images to video, 1 for yes, 0 for no
- --vis_mode: display mode. 1 is to show inference+annotated point cloud, -1 is not show.
- --data_processed: whether the data has been annotated and predicted, 1 for yes, 0 for no.

## raw data w/o annotation and prediction, visualize raw data only: 
`python data_check.py --path /home/zx/Desktop/zx/TAP3D_demo/data/test1 --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --thermal_input m08 --use_old_plot 1 --no_id_distinguish 0 --img2vid 0 --vis_mode -1 --data_processed 0`

## visualize data with annotation and prediction, vidualizing raw data + annotated point cloud gt + prediction:
`python data_check.py --path /home/zx/Desktop/zx/TAP3D_demo/data/test1 --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --thermal_input m08 --use_old_plot 1 --no_id_distinguish 0 --img2vid 0 --vis_mode 1 --data_processed 1`

## check data:
`python data_check.py --path /home/zx/Desktop/zx/TAP3D_demo/data/test_multi --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --thermal_input m08 --use_old_plot 0 --no_id_distinguish 0 --img2vid 0 --vis_mode -1 --data_processed 0`


## annotating data
To annotate data, do `python DataAnnotation.py --sensor_name senxor_m08 --visualization_flag 1 --raw_data_folder data`

## collect data with timestamp as dir name under data/:
`python data_collection.py --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --collection_duration 1200 --save 1`

# A temporary patch:
`python data_collection_single.py --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth --collection_duration 1200 --thermal_input m08  --save 0 --inference 1 --use_old_plot 0`

python data_collection_single.py --exp_config_file model3_m16 --weights weights/m16/model3_m16_thermo_pt_0820152400.pth --collection_duration 1200 --thermal_input m16  --save 0 --inference 1 --use_old_plot 0

TODO
2. gt pcd compare
3. point cloud gt
4. gt for data check
