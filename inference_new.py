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

from Models import *
# from ThermalDataset import ThermalMeshDataset
from typing import Optional, Callable, Any, Dict, Tuple
import torchvision.models as models
from depth2ptcloud import DepthMask2PointCloudFast, plot_3d_point_cloud
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


torch.manual_seed(42)
print("training: seed is fixed")

class M08ToPtcloud():
    def __init__(self, config_folder, config_file_name, weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_config = self.get_config(config_folder, config_file_name)
        self.model = self.get_model()
        if weights is not None:
            state = torch.load(weights, map_location=self.device)
            # Allow partial load for fine-tuned heads etc.
            try:
                self.model.load_state_dict(state, strict=False)
            except Exception:
                self.model.load_state_dict(state)
            print("Loaded weights from:", weights)
        self.d2p = DepthMask2PointCloudFast(self.exp_config)

    def get_config(self, config_folder, config_file_name):
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

    def get_model(self):
        
        model_name = self.exp_config['model_name']
        model = get_registered_models(model_name, self.exp_config)
        model.to(self.device)
        print("Model loaded: ", model_name)
        model.eval()
        return model
    
    # excute the model and get the predictions and the labels
    def thermal2depth(self, thermal_images):
        # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient computation during evaluation
            # for i, sample in enumerate(tqdm(data_loader, disable= config['tqdm_disable'])):
            # thermal_images, vertices = sample
            

            thermal_images = thermal_images.float().to(self.device)   
            predict = self.model(thermal_images)

            depth = predict['refined_depth']
            indicator = predict['user_index']
            forground_background_mask = (indicator > 0.5).float()
            depthPred = torch.cat([depth, indicator, forground_background_mask], dim=1)
            
        return depthPred

    def depth2ptcloud(self, depth):
        ptcloud = self.d2p(depth)[0]
        return ptcloud
    
    def thermal2ptcloud(self, thermal):
        depth = self.thermal2depth(thermal)
        ptcloud = self.depth2ptcloud(depth)
        return ptcloud

if __name__ == "__main__":
    #python inference_new.py --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth
    parser = argparse.ArgumentParser(description='Infer')
    parser.add_argument("--exp_config_file", type=str, help="Configuration YAML file of the experiment")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth weights (optional)")
    parser.add_argument("--train", type=int, default="0", help="0 is test, 1 is train")
    args = parser.parse_args()
    
    # loading the configuration file ########################################
    exp_config_file_name = args.exp_config_file + '.yaml'
    t2p = M08ToPtcloud('exp_configs', exp_config_file_name, args.weights)
    
    thermal_image = np.load("testm08.npy")
    thermal_images = np.expand_dims(thermal_image, axis=0)
    thermal_images = np.expand_dims(thermal_images, axis=0)
    thermal_images = torch.from_numpy(thermal_images)
    print("thermal img:", thermal_images.shape)

    # Perform inference on the test dataset
    # predicteddepth = t2p.thermal2depth(thermal_images)
    # np.save("results.npy", predicteddepth.cpu().numpy())

    # directly predict ptcloud
    ptcloud = t2p.thermal2ptcloud(thermal_images)
    print("shape of pt cloud:", ptcloud.shape)
    plot_3d_point_cloud(ptcloud.cpu().numpy(), 6, 1000)
    plt.savefig("resultsPCD.pdf")

    
    