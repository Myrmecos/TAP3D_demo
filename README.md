# Running demo of TAP3D

- step 0. move model to `weights/m08`

- step 1. plug in m08 and realsense camera

- step 2. run the demo program: `python data_collection.py --exp_config_file model3_m08 --weights weights/m08/Demo_m08_thermo_pt_0826180626.pth --collection_duration 1200`. 

    - --exp_config_file: the config file for the model, under directory `exp_configs/`

    - --weights: the path to the model weights, under `weights/m08/` directory

    - --collection_duration: the duration of data collection in seconds, default is 60s (1min)


# saving data
To save data to data/ folder, do: `python data_collection.py --exp_config_file model3_m08 --weights weights/m08/Demo_m08_thermo_pt_0826180626.pth --collection_duration 1200 --save 1 --save_dest data/entry0`


# annotating data
To annotate data, do `python DataAnnotation.py --sensor_name senxor_m08 --visualization_flag 1 --raw_data_folder data`