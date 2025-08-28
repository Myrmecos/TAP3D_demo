import cv2
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset,random_split
import yaml
import os
import argparse
import torch
from tqdm import tqdm
import math
import pickle
from torch.utils.tensorboard import SummaryWriter
import os
import timm
from timm.scheduler import CosineLRScheduler
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from Models import *
# from ThermalDataset import ThermalMeshDataset
from typing import Optional, Callable, Any, Dict, Tuple
import torchvision.models as models
from depth2ptcloud import DepthMask2PointCloudFast, plot_3d_point_cloud
import matplotlib.pyplot as plt

torch.manual_seed(42)
print("training: seed is fixed")


def get_config(config_folder, config_file_name):
    if '.yaml' not in config_file_name:
        config_file_name = config_file_name + '.yaml'

    all_yaml_files = []
    all_yaml_file_paths = []
    for root, dirs, files in os.walk(config_folder):
        for file in files:
            if file.endswith(".yaml"):
                all_yaml_files.append(file)
                all_yaml_file_paths.append(os.path.join(root, file))

    if config_file_name not in all_yaml_files:
        print("Configuration file name is: ",config_file_name)
        print("The configuration file is not found! Please check the file name!")
        exit
    else:
        # if there are multiple configuration files with the same name, print all the locations
        if all_yaml_files.count(config_file_name) > 1:
            print("The configuration file is: ",config_file_name)
            print("There are multiple configuration files with the same name!")
            for i in range(all_yaml_files.count(config_file_name)):
                print(all_yaml_file_paths[all_yaml_files.index(config_file_name, i)])
            exit
        else:
            config_file_path = all_yaml_file_paths[all_yaml_files.index(config_file_name)]
            print("The configuration file is found at: ",config_file_path)
            with open(config_file_path, 'r') as fd:
                config = yaml.load(fd, Loader=yaml.FullLoader)
    return config



# excute the model and get the predictions and the labels
def inference(model, thermal_images, config, device):
    # Set the model to evaluation mode

    loss_all = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        # for i, sample in enumerate(tqdm(data_loader, disable= config['tqdm_disable'])):
        # thermal_images, vertices = sample
        

        thermal_images = thermal_images.float().to(device)   
        predict = model(thermal_images)

        depth = predict['refined_depth']
        indicator = predict['user_index']
        forground_background_mask = (indicator > 0.5).float()
        depthPred = torch.cat([depth, indicator, forground_background_mask], dim=1)

    return depthPred

def depth2ptcloud(depth):
    ptcloud = None
    return ptcloud

if __name__ == "__main__":
    #python inference_ptcloud_only.py --exp_config_file model3_m08 --weights /home/shared/smplxTest/weights/m08/model3_m08_thermo_pt_0819203728.pth
    parser = argparse.ArgumentParser(description='Infer')
    parser.add_argument("--exp_config_file", type=str, help="Configuration YAML file of the experiment")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth weights (optional)")
    parser.add_argument("--train", type=int, default="0", help="0 is test, 1 is train")
    args = parser.parse_args()
    
    # loading the configuration file ########################################
    exp_config_file_name = args.exp_config_file + '.yaml'
    exp_config = get_config('exp_configs', exp_config_file_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = exp_config['model_name']
    model = get_registered_models(model_name, exp_config)
    model.to(device)
    print("Model loaded: ", model_name)
    model.eval()  

    # prepare depth2ptcloud class
    d2p = DepthMask2PointCloudFast(exp_config)

    if args.weights is not None:
        state = torch.load(args.weights, map_location=device)
        # Allow partial load for fine-tuned heads etc.
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            model.load_state_dict(state)
        print("Loaded weights from:", args.weights)
    
    thermal_image = np.load("../smplxTest/data/U0_E0_1_sitting_1o3_none_0/senxor_m08/1739524850.6177948.npy")
    thermal_images = np.expand_dims(thermal_image, axis=0)
    thermal_images = np.expand_dims(thermal_images, axis=0)
    thermal_images = torch.from_numpy(thermal_images)
    
    print(thermal_images.shape)

    # Perform inference on the test dataset
    predicteddepth = inference(model, thermal_images, exp_config, device)
    
    np.save("results.npy", predicteddepth.cpu().numpy())
    ptcloud = d2p(predicteddepth)[0]
    print("shape of pt cloud:", ptcloud.shape)
    plot_3d_point_cloud(ptcloud.cpu().numpy(), 6, 1000)
    plt.savefig("resultsPCD.pdf")

    
    