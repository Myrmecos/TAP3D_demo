import serial
import time
import ast

import yaml
import numpy as np
import cv2
import sys
import os
import signal
import logging
import cv2 as cv
from pprint import pprint
import argparse
import pyrealsense2 as rs
import copy
from inference_new import M08ToPtcloud
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
matplotlib.use('Agg')  # Ensure Agg backend is set
import matplotlib.pyplot as plt
import logging
from DataAnnotation import DataAnnotate
import pickle as pkl
from plot import plot_3d_point_cloud_new, remove_small_regions, mark_connected_components
cnt = 0
logging.getLogger().setLevel(logging.CRITICAL)
# sys.path.append("/home/zx/Desktop/zx/DeepTadarDataCollect-ubuntu-data-collect/")

colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray']

import seekcamera
from seekcamera import (
    SeekCameraIOType,
    SeekCameraColorPalette,
    SeekCameraManager,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
    SeekCamera,
    SeekFrame,
)
from collections import deque
import threading
import pickle
from time import sleep
from threading import Condition
# from senxor.utils import connect_senxor, data_to_frame, remap
# from senxor.utils import cv_filter, cv_render, RollingAverageFilter
import senxor
import senxor.utils
import senxor_previous
import senxor_previous.utils

def put_temp(image, temp1, temp2, sensor_name):
    cv2.putText(image, f"{sensor_name}: {temp1:.1f}~{temp2:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(image, f"{sensor_name}: {temp1:.1f}~{temp2:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def put_text(img, text):
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
class MLXSensor:
    def __init__(self, sensor_port):
        self.sensor_port = sensor_port
        self.ser = serial.Serial(self.sensor_port, 921600, timeout=1)

    def read_data(self):
        data = self.ser.readline().strip()
        if len(data) > 0:
            try:
                msg_str = str(data.decode('utf-8'))
                msg = ast.literal_eval(msg_str)
                return msg
            except:
                return None
        return None

    def get_temperature_map(self):
        data = self.read_data()
        if data is not None:
            temp = np.array(data["temperature"]) # 768
            if len(temp) == 768:
                temp = temp.reshape(24, 32)
                return temp
        return None

    def get_ambient_temperature(self):
        data = self.read_data()
        if data:
            return data["at"]
        return None

    def close(self):
        self.ser.close()

    def SubpageInterpolating(self,subpage):
        shape = subpage.shape
        mat = subpage.copy()
        for i in range(shape[0]):
            for j in range(shape[1]):
                if mat[i,j] > 0.0:
                    continue
                num = 0
                try:
                    top = mat[i-1,j]
                    num = num+1
                except:
                    top = 0.0

                try:
                    down = mat[i+1,j]
                    num = num+1
                except:
                    down = 0.0

                try:
                    left = mat[i,j-1]
                    num = num+1
                except:
                    left = 0.0

                try:
                    right = mat[i,j+1]
                    num = num+1
                except:
                    right = 0.0
                mat[i,j] = (top + down + left + right)/num
        return mat


class senxor_16:
    def __init__(self, sensor_port = "/dev/ttyACM0"):
        self.sensor_port = sensor_port
        self.mi48 = senxor.utils.connect_senxor(comport=self.sensor_port)
        self.setup_thermal_camera(fps_divisor=3)

        self.mi48.set_data_type('temperature')
        self.mi48.set_temperature_units('Celsius')

        self.ncols, self.nrows = self.mi48.fpa_shape
        self.mi48.start(stream=True, with_header=True)

    def get_temperature_map(self):
        return self.mi48.read() # data, header

    def get_temperature_map_shape(self):
        return self.ncols, self.nrows

    def setup_thermal_camera(self, fps_divisor = 3):
        self.mi48.regwrite(0xB4, fps_divisor)  #
        # Disable firmware filters and min/max stabilisation
        # no FW filtering for Panther in the mi48 for the moment
        # self.mi48.regwrite(0xD0, 0x00)  # temporal
        # self.mi48.regwrite(0x20, 0x00)  # stark
        # self.mi48.regwrite(0x25, 0x00)  # MMS
        # self.mi48.regwrite(0x30, 0x00)  # median

        self.mi48.regwrite(0xD0, 0x00)  # temporal
        self.mi48.regwrite(0x30, 0x00)  # median
        self.mi48.regwrite(0x20, 0x03)  # stark
        self.mi48.regwrite(0x25, 0x01)  # MMS
        self.mi48.set_fps(30)
        self.mi48.set_emissivity(0.95)  # emissivity to 0.95, as used in calibration,
                                       # so there is no sensitivity change
        self.mi48.set_sens_factor(1.0)  # sensitivity factor 1.0
        self.mi48.set_offset_corr(0.0)  # offset 0.0
        self.mi48.set_otf(0.0)          # otf = 0
        self.mi48.regwrite(0x02, 0x00)  # disable readout error compensation

    def close(self):
        self.mi48.stop()

class senxor_08:
    def __init__(self, sensor_port = "/dev/ttyACM0"):
        self.sensor_port = sensor_port
        self.mi48 = senxor_previous.utils.connect_senxor(src=self.sensor_port)
        self.setup_thermal_camera(fps_divisor=3)

        self.mi48.set_data_type('temperature')
        self.mi48.set_temperature_units('Celsius')

        self.ncols, self.nrows = self.mi48.fpa_shape
        self.mi48.start(stream=True, with_header=True)

    def get_temperature_map(self):
        return self.mi48.read() # data, header

    def get_temperature_map_shape(self):
        return self.ncols, self.nrows

    def setup_thermal_camera(self, fps_divisor = 3):
        self.mi48.regwrite(0xB4, fps_divisor)  #
        # MMS and STARK are sufficient for Cougar
        # self.mi48.disable_filter(f1=True, f2=True, f3=True)
        self.mi48.regwrite(0xD0, 0x00)  # temporal
        self.mi48.regwrite(0x30, 0x00)  # median
        self.mi48.regwrite(0x20, 0x03)  # stark
        self.mi48.regwrite(0x25, 0x01)  # MMS
        self.mi48.set_fps(30)
        self.mi48.set_emissivity(0.95)  # emissivity to 0.95, as used in calibration,
                                       # so there is no sensitivity change
        self.mi48.set_sens_factor(1.0)  # sensitivity factor 1.0
        self.mi48.set_offset_corr(0.0)  # offset 0.0
        self.mi48.set_otf(0.0)          # otf = 0
        self.mi48.regwrite(0x02, 0x00)  # disable readout error compensation

    def close(self):
        self.mi48.stop()


class senxor_postprocess:
    def __init__(self):
        # set cv_filter parameters
        self.par = {'blur_ks':3, 'd':5, 'sigmaColor': 27, 'sigmaSpace': 27}
        self.dminav = senxor_previous.utils.RollingAverageFilter(N=10)
        self.dmaxav = senxor_previous.utils.RollingAverageFilter(N=10)

    def process_temperature_map(self, data):
        min_temp = self.dminav(data.min())  # + 1.5
        max_temp = self.dmaxav(data.max())  # - 1.5
        frame = np.clip(data, min_temp, max_temp)
        filt_uint8 = senxor_previous.utils.cv_filter(senxor_previous.utils.remap(frame), self.par, use_median=True,
                           use_bilat=True, use_nlm=False)
        return filt_uint8


class realsense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(self.config)
        #below for testing only ====
        # device = profile.get_device()
        # device.hardware_reset()
        #above for testing only ====
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image


class Renderer:
    """Contains camera and image data required to render images to the screen."""
    def __init__(self):
        self.busy = False
        self.frame = SeekFrame()
        self.camera = SeekCamera()
        self.frame_condition = Condition()
        self.first_frame = True

def argb2bgr(frame):
    """Converts an RGBA8888 frame to a BGR frame."""
    if frame.shape[2] != 4:
        raise ValueError("Input frame must be RGBA8888")
    bgr_image = frame[:, :, 1:][:, :, ::-1]
    return bgr_image

class seekthermal:
    def __init__(self, data_format="color"):
        self.data_format = data_format
        self.manager = SeekCameraManager(SeekCameraIOType.USB)
        if self.data_format == "color":
            self.renderer = Renderer()
            self.manager.register_event_callback(self._on_event, self.renderer)
            self.frame_condition = Condition()
        else:
            self.data_frame = None
            self.data_condition = False
            def on_frame2(camera, camera_frame, file):
                frame = camera_frame.thermography_float
                self.data_frame = frame.data
                self.data_condition = True
                # sleep(0.1)
            def on_event2(camera, event_type, event_status, user_data):
                print("{}: {}".format(str(event_type), camera.chipid))

                if event_type == SeekCameraManagerEvent.CONNECT:
                    camera.register_frame_available_callback(on_frame2, None)
                    camera.capture_session_start(SeekCameraFrameFormat.THERMOGRAPHY_FLOAT)

            self.manager.register_event_callback(on_event2)

    def _on_event(self, camera, event_type, event_status, renderer):
        print("{}: {}".format(str(event_type), camera.chipid))

        def on_frame(_camera, camera_frame, renderer):
            with renderer.frame_condition:
                renderer.frame = camera_frame.color_argb8888
                renderer.frame_condition.notify()

        if event_type == SeekCameraManagerEvent.CONNECT:
            if renderer.busy:
                return
            renderer.busy = True
            renderer.camera = camera
            renderer.first_frame = True
            camera.color_palette = SeekCameraColorPalette.TYRIAN
            camera.register_frame_available_callback(on_frame, renderer)
            camera.capture_session_start(SeekCameraFrameFormat.COLOR_ARGB8888)


    def get_frame(self):
        if self.data_format == "color":
            with self.renderer.frame_condition:
                if self.renderer.frame_condition.wait(150.0 / 1000.0):
                    frame = self.renderer.frame.data
                    if frame is not None:
                        return frame
        else:
            #print(self.data_frame)
            return self.data_frame
        return None

    def close(self):
        try:
            self.renderer.camera.capture_session_stop()
        except:
            pass
        self.manager.destroy()

class image_buffer():
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.read = 0
        self.write = 0
        self.buffer = []
        for i in range (buffer_size):
            self.buffer.append(None)


    def add(self, image):
        #if self.buffer[self.write] is not None:
        self.buffer[self.write] = image
        self.write += 1
        self.write = self.write%self.buffer_size


    def get(self):
        self.read += 1
        self.read %= self.buffer_size
        return self.buffer[self.read]

def plot_3d_point_cloud(fig, ax, point_cloud, max_num_persons = 0, max_num_points = 0, no_id_distinguish = False, threshold=0.1, s= 10, regularSpacing = True):
    points_per_person = max_num_points + 1
    scatter_ret = None


    # Define colormap for different users
    def plot_camera(ax):
        camera_vertices = np.array([
            [-100, 100, -100], [100, 100, -100], [100, 100, 60], [-100, 100, 60],
            [0, -100, 0], [0, -100, 0], [0, -100, 0], [0, -100, 0]
        ])
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        rotated_vertices = camera_vertices @ rotation_matrix.T
        translated_vertices = rotated_vertices + np.array([0, 0, 0])
        camera_faces = [
            [translated_vertices[0], translated_vertices[1], translated_vertices[2], translated_vertices[3]],
            [translated_vertices[0], translated_vertices[1], translated_vertices[5], translated_vertices[4]],
            [translated_vertices[2], translated_vertices[3], translated_vertices[7], translated_vertices[6]],
            [translated_vertices[1], translated_vertices[2], translated_vertices[6], translated_vertices[5]],
            [translated_vertices[0], translated_vertices[3], translated_vertices[7], translated_vertices[4]]
        ]
        plot_faces = []
        for face in camera_faces:
            plot_face = []
            for vert in face:
                plot_face.append([vert[0], vert[2], -vert[1]])
            plot_faces.append(plot_face)
        ax.add_collection3d(Poly3DCollection(
            verts=plot_faces,
            facecolors='gray',
            linewidths=1,
            edgecolors='black',
            alpha=1
        ))

    print("!!!!!!!!!!!!!DEBUG: no_id_distinguish:", no_id_distinguish)
    plot_camera(ax)
    global colors
    if no_id_distinguish:
        colors = ['red']*6
        
    # Plot points for each person
    if regularSpacing:
        for person_idx in range(max_num_persons):
            # Extract points for this person (assuming each person has max_num_points)
            start_idx = person_idx * points_per_person
            end_idx = start_idx + points_per_person

            indicator_idx = (person_idx + 1) * points_per_person - 1
            indicator_point = point_cloud[0, indicator_idx]
            if indicator_point > threshold:
                # Get points for this person
                person_points = point_cloud[ :, start_idx:end_idx]

                # Reshape to get individual 3D points
                x = person_points[0, :]
                y = person_points[1, :]
                z = person_points[2, :]

                # Filter out points where all coordinates are 0
                valid_points = ~((x < 5) & (y < 5) & (z < 5) & (x > -5) & (y > -5) & (z > -5))
                x_valid = x[valid_points]
                y_valid = y[valid_points]
                y_valid = -y_valid
                z_valid = z[valid_points]

                if len(x_valid) > 0:  # Only plot if there are valid points
                    scatter_ret = ax.scatter(x_valid, z_valid, y_valid,
                            label="", alpha=0.5, s=s, c=colors[person_idx])
    else:
        for person_idx, person_points in enumerate(point_cloud):
            # Reshape to get individual 3D points
            person_points = person_points.T
            x = person_points[0, :]
            y = person_points[1, :]
            z = person_points[2, :]

            # Filter out points where all coordinates are 0
            valid_points = ~((x < 5) & (y < 5) & (z < 5) & (x > -5) & (y > -5) & (z > -5))
            x_valid = x[valid_points]
            y_valid = y[valid_points]
            y_valid = -y_valid
            z_valid = z[valid_points]

            if len(x_valid) > 0:  # Only plot if there are valid points
                scatter_ret = ax.scatter(x_valid, z_valid, y_valid,
                        label="", alpha=0.5, s=s, c=colors[person_idx])
    ax.set_xlim([-2000, 2000])
    ax.set_ylim([0, 4000])
    ax.set_zlim([-1000, 1000])
    # how can I change the viewing angle?
    # ans:
    # ax.view_init(elev=20, azim=-0)

def process_mask(result_dict):
    '''
    Make a mask on white background
    each human is assigned a color
    '''
    print("DEBUG: PROCESSING MASK!!!!!!!")
    mask = None
    
    pcl_gt =  result_dict['depth_mask_person']
    pcl_dist = result_dict['depth_person']
    
    # sort pcl_gt's element according to pcl_dist
    indices = np.argsort(pcl_dist)
    # pcl_gt = [pcl_gt[i] for i in indices]
        

    # Get colors for each person
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    color_map = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128)
    }
    
    for i in range(result_dict['num_persons']):
        target = pcl_gt[indices[i]]
        if mask is None:
            # make a white mask with shape same as result_dict['depth_mask_person']
            mask = np.zeros_like(target)
            mask[:] = 255
            # repeat to 3 channels
            mask = np.stack([mask] * 3, axis=-1)

        mask[target > 0] = color_map[colors[i]] # assign the i-th color in the global color array (which contains strings of color). we need only a value. mask and target are both 2d arrays
        
        

    if mask is None:
        mask = np.ones((240, 320, 3), dtype=np.uint8)*255
        
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (320, 240), interpolation=cv2.INTER_NEAREST)
    # mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # write on the top-left of the mask: "human mask"
    put_text(mask, "human_mask")

    return mask

# results_dict = {
#     'num_persons': 0,
#     'depth_person': [],
#     'depth_mask_person': [],  
#     'point_cloud_person': [],
#     '2D_pose_person': [],
# }
def concat_pcd(result_dict):
    # Concatenate point clouds for all persons
    # point cloud shape: (N, 3)
    if not result_dict['point_cloud_person']:
        return None
    return np.concatenate(result_dict['point_cloud_person'], axis=0)

def process_depth(result_dict, h = 62, w = 80):
    # obtain a shape [1, 3, 62, 80] tensor. the 3 chanels represent depth, user index and foreground-background mask.
    depth = result_dict['depth_person'][0] if result_dict['depth_person'] else np.zeros((h, w), dtype=np.float32)
    user_index = result_dict['user_index'][0] if result_dict['user_index'] else np.zeros((h, w), dtype=np.float32)
    fg_bg_mask = result_dict['fg_bg_mask'][0] if result_dict['fg_bg_mask'] else np.zeros((h, w), dtype=np.float32)

    depth_3channel = torch.cat([depth, user_index, fg_bg_mask], dim=1)
    return depth_3channel

class DataProcessor:
    def __init__(self):
        pass
    
     # ================================== check exist and processing data ==================================
    def process_data_raw(self, realsense_depth_image, realsense_color_image, temp_ori):
        if realsense_depth_image is None or realsense_color_image is None or temp_ori is None or seek_camera_frame is None:
            return None
        
        # print("DEBUG: shape of image:", num_cols_m08, num_rows_m08)
        # print("DEBUG: shape of m16:", num_cols_m16, num_rows_m16)
        
        # preprocess frames: organize pixels and orientation
        senxor_temperature_map_m08 = senxor_temperature_map_m08.reshape(num_cols_m08, num_rows_m08)
        senxor_temperature_map_m08 = np.flip(senxor_temperature_map_m08, 0)
        senxor_temperature_map_m16 = senxor_temperature_map_m16.reshape(num_cols_m16, num_rows_m16)
        senxor_temperature_map_m16 = np.flip(senxor_temperature_map_m16, 0)
        seek_camera_frame = np.flip(seek_camera_frame, 0)
        seek_camera_frame = np.flip(seek_camera_frame, 1)
        
        # postprocess
        if args.mi08_process:
            senxor_temperature_map_m08 = senxor_postprocess_m.process_temperature_map(senxor_temperature_map_m08)
        if args.mi16_process:
            senxor_temperature_map_m16 = senxor_postprocess_m.process_temperature_map(senxor_temperature_map_m16)

        return [realsense_depth_image,
            realsense_color_image,
            senxor_temperature_map_m08,
            senxor_temperature_map_m16,
            seek_camera_frame
        ]









    # ================================== saving the raw data ==================================
    def get_timestampstr():
        return time.strftime("%Y%m%d%H%M%S", time.localtime()) + f"{int((time.time()%1)*1e6):06d}"
    
    # preparing file name
    def save_raw_data(self, realsense_depth_image,
                      realsense_color_image, 
                      thermal_map, 
                      timestampstr):
        
        npyname = timestampstr + ".npy"
        pklname = timestampstr + ".pkl"
        
        # timestamp format: yyyymmddhhmmssffffff
        imgpath = os.path.join(imgdest, npyname)
        thermal_path = os.path.join(thermal_dest, npyname)
        depthpath = os.path.join(depthdest, npyname)
        
        # depthoutputpath = os.path.join(depthoutputdest, npyname)
        np.save(imgpath, realsense_color_image)
        np.save(depthpath, realsense_depth_image)
        np.save(thermal_path, thermal_map)

            



    # process depth mask, to re-assign id or others
    def process_depth(self, depth_ori, no_id=False):
        '''
        Take a depth map of shape (1, 3, x, y)
        returns a depth map of shape (1, 3, x, y) that is cleaned
        no_id: all human assigned an id of 1
        '''
        depth = depth_ori.cpu().numpy()
        depth, indicator, foreground_background_mask = depth[0, 0], depth[0, 1], depth[0, 2]
        print(depth.shape, indicator.shape, foreground_background_mask.shape, "DEBUG: all, before removing small regions")
        
        foreground_background_mask = remove_small_regions(foreground_background_mask)
        print(depth.shape, indicator.shape, foreground_background_mask.shape, "DEBUG: all, after removing small regions")
        depth[~foreground_background_mask] = 0
        if no_id:
            indicator = foreground_background_mask
        else:
            # indicator[~foreground_background_mask] = 0
            indicator = mark_connected_components(foreground_background_mask)
            # indicator now contains 0 for background, 1 for person 1, 2 for person 2...
            # now we want to re-assign the ids according to the depth. The closer person will be assigned a smaller id.
            unique_ids = np.unique(indicator)
            depth_lst = []
            for uid in unique_ids:
                if uid == 0:
                    continue
                # find the pixels belonging to this uid
                mask = (indicator == uid)
                # find the minimum depth value among these pixels
                mean_depth = depth[mask].mean()
                # assign this min depth value to the corresponding pixels in the indicator
                depth_lst.append([mean_depth, uid])
            depth_lst.sort()  # sort by depth, ascending
            
            new_indicator = np.zeros_like(indicator)
            for i, (d, uid) in enumerate(depth_lst):
                new_indicator[indicator == uid] = i + 1
            indicator = new_indicator

        # # round indicator values to nearest int
        # indicator = np.round(indicator).astype(np.int32)

        print(depth.shape, indicator.shape, foreground_background_mask.shape, "DEBUG: all, after removing small regions")
        depth = torch.from_numpy(depth).float()  # (1, 1, H, W)
        indicator = torch.from_numpy(indicator).float()
        foreground_background_mask = torch.from_numpy(foreground_background_mask.copy())
        
    
        # Move back to original device
        device = depth_ori.device
        depth = depth.to(device).unsqueeze(0).unsqueeze(0)
        indicator = indicator.to(device).unsqueeze(0).unsqueeze(0)
        foreground_background_mask = foreground_background_mask.to(device).unsqueeze(0).unsqueeze(0)

        res = torch.cat([depth, indicator, foreground_background_mask], dim=1)
        
        # global cnt
        # if cnt == 10:
        #     time.sleep(1)
        #     np.save("depth.npy", depth.cpu().numpy()[0, 0])
        #     np.save("indicator.npy", indicator.cpu().numpy()[0, 0])
        #     np.save("forground_background_mask.npy", forground_background_mask.cpu().numpy()[0, 0])
        #     exit(0)
            
        # cnt += 1
        return res
        

    # ================================== Inference: get the predicted point clouds ================================================
    def get_point_clouds_pred(self, t2p, thermal_image):
            
        # MODEL calling
        thermal_image = np.expand_dims(thermal_image, axis=0)
        thermal_image = np.expand_dims(thermal_image, axis=0)
        thermal_image = torch.from_numpy(thermal_image.copy())
        
        # produce point cloud visualization for m08
        # ptcloud = t2p.thermal2ptcloud(thermal_images)
        print("shape of thermal image: ", thermal_image.shape)
        depth = t2p.thermal2depth(thermal_image)
        # depth: depth, nidicator, foreground_background_mask
        depth = self.process_depth(depth)
        ptcloud = t2p.depth2ptcloud(depth)
        return ptcloud.cpu().numpy()

    def get_annotation(self, realsense_color_image, realsense_depth_image, annotator):
        # Annotate: get annotation dictionary +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        result_dict = annotator.forward(realsense_color_image, realsense_depth_image)
        return result_dict

    def save_pcd_pred(self, ptcloud, timestampstr, pointcloudoutputdest):
        # Inference: save predicted point clouds ============================================================
        npyname = timestampstr + ".npy"
        pklname = timestampstr + ".pkl"
        # save point cloud ptcloud
        pointcloudpath = os.path.join(pointcloudoutputdest, npyname)
        np.save(pointcloudpath, ptcloud)

    def save_annotation(self, result_dict, timestampstr, annotationdest):
        # Inference: save annotated point clouds ============================================================
        npyname = timestampstr + ".npy"
        pklname = timestampstr + ".pkl"
        # save annotation
        annotationpath = os.path.join(annotationdest, pklname)
        pkl.dump(result_dict, open(annotationpath, "wb"))

        
        
        
        
        
        
        
    def visualize_gt_pcd(self, fig, ax, result_dict, no_id_distinguish, use_old_plot = False):
        print("#####visualize_gt_pcd: no_id-distinguish:", no_id_distinguish)
        
        pcl_gt =  result_dict['point_cloud_person']
        pcl_dist = result_dict['depth_person']
        # sort pcl_gt's element according to pcl_dist
        indices = np.argsort(pcl_dist)
        print(len(pcl_gt), len(indices), "DEBUG: length of pcl_gt and pcl_dist")
        pcl_gt = [pcl_gt[i] for i in indices]
        
        if use_old_plot:
            ax.clear()
            if pcl_gt is not None:
                # print("DEBUG: shape is:", pcl_gt.shape)
                print("visualize_gt_pcd, before calling plot_3d_pcd:")
                plot_3d_point_cloud(fig, ax, pcl_gt, no_id_distinguish=no_id_distinguish, regularSpacing=False)
            else:
                plot_3d_point_cloud(fig, ax, np.zeros([[3, 1*42]]), no_id_distinguish=no_id_distinguish, regularSpacing=False)
            fig.canvas.draw()
            # fig.canvas.flush_events()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (960, int(960 * image.shape[0] / image.shape[1])))
        else:
            labels1 = [f'Pred. P{i+1}' for i in range(6)]
            colors1 = plt.colormaps.get_cmap('Set1')(np.linspace(0, 1, 6))
            if pcl_gt is None:
                pcl_gt = np.zeros([1*6006, 3])
            image = plot_3d_point_cloud_new(pcl_gt, len(pcl_gt), -1, camera_height=1.3, labels=labels1, colors=colors1, no_id_distinguish=no_id_distinguish, regularSpacing=False)
            image = cv2.resize(image, (960, int(960 * image.shape[0] / image.shape[1])))
            put_text(image, "Ground Truth")
        return image

        
    # ================== for visualization of point clouds: 2 axes for inference and annotate, 1 axis for annotate, no axis for collection ============
    def visualize_pred_pcd(self, fig, ax1, ptcloud, exp_config, no_id_distinguish, use_old_plot = False):
        print("?????????????DEBUG: use_old_plot:", use_old_plot)
        if use_old_plot:
            ax1.clear()
            # print(ptcloud.cpu().numpy().shape, "DDDDEBUG")
            plot_3d_point_cloud(fig, ax1, ptcloud,  exp_config['max_num_persons'], exp_config['max_num_points'], no_id_distinguish=no_id_distinguish)

            fig.canvas.draw()
            # fig.canvas.flush_events()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # rescale image such that its width is 960, and its height-width ration remains unchanged
            # cv2.putText(image, f"Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # cv2.putText(image, f"Prediction", (10 + 960, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            image = cv2.resize(image, (960, int(960 * image.shape[0] / image.shape[1])))
        else:
            labels1 = [f'Pred. P{i+1}' for i in range(6)]
            colors1 = plt.colormaps.get_cmap('Set1')(np.linspace(0, 1, 6))
            image = plot_3d_point_cloud_new(ptcloud,  exp_config['max_num_persons'], exp_config['max_num_points'], camera_height=1.3, labels=labels1, colors=colors1, no_id_distinguish=no_id_distinguish)
            image = cv2.resize(image, (960, int(960 * image.shape[0] / image.shape[1])))
            put_text(image, "Prediction")
        print("DEBUG: image shapeeeee:", image.shape)
        return image


    def prepare_sensor_visuals(self, realsense_color_image, realsense_depth_image, senxor_temperature_map_m08, senxor_temperature_map_m16, seek_camera_frame, point_cloud_image, mask, inference_mode):  
        # ================================== Prepare the images for visualization ==================================
        # visualize realsense
        realsense_depth_image = cv2.applyColorMap(cv2.convertScaleAbs(realsense_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        realsense_depth_image = cv2.resize(realsense_depth_image, (320, 240))
        realsense_color_image = cv2.resize(realsense_color_image, (320, 240), interpolation=cv2.INTER_NEAREST)

        # visualize m08
        m08_min = -1024
        m08_max = -1024
        m08_min = np.min(senxor_temperature_map_m08)
        m08_max = np.max(senxor_temperature_map_m08)
        senxor_temperature_map_m08 = senxor_temperature_map_m08.astype(np.uint8)
        senxor_temperature_map_m08 = cv2.normalize(senxor_temperature_map_m08, None, 0, 255, cv2.NORM_MINMAX)
        senxor_temperature_map_m08 = cv2.resize(senxor_temperature_map_m08, (320, 240), interpolation=cv2.INTER_NEAREST)
        senxor_temperature_map_m08 = cv2.applyColorMap(senxor_temperature_map_m08, cv2.COLORMAP_JET)
        put_temp(senxor_temperature_map_m08, m08_min, m08_max, "m08")
        
        m16_min = -1024
        m16_max = -1024
        m16_min = np.min(senxor_temperature_map_m16)
        m16_max = np.max(senxor_temperature_map_m16)
        senxor_temperature_map_m16 = senxor_temperature_map_m16.astype(np.uint8)
        senxor_temperature_map_m16 = cv2.normalize(senxor_temperature_map_m16, None, 0, 255, cv2.NORM_MINMAX)
        senxor_temperature_map_m16 = cv2.resize(senxor_temperature_map_m16, (320, 240), interpolation=cv2.INTER_NEAREST)
        senxor_temperature_map_m16 = cv2.applyColorMap(senxor_temperature_map_m16, cv2.COLORMAP_JET)
        put_temp(senxor_temperature_map_m16, m16_min, m16_max, "m16")

        # visualize seek camera
        seek_min = -1024
        seek_max = -1024
        seek_min = np.min(seek_camera_frame)
        seek_max = np.max(seek_camera_frame)
        seek_camera_frame = seek_camera_frame.astype(np.uint8)
        seek_camera_frame = cv2.normalize(seek_camera_frame, None, 0, 255, cv2.NORM_MINMAX)
        seek_camera_frame = cv2.resize(seek_camera_frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        seek_camera_frame = cv2.applyColorMap(seek_camera_frame, cv2.COLORMAP_JET)
        put_temp(seek_camera_frame, seek_min, seek_max, "seek")
        
        # put text on depth and color
        put_text(realsense_depth_image, "realsense depth")
        put_text(realsense_color_image, "realsense color")

        #print(realsense_depth_image.shape, realsense_color_image.shape, seek_camera_frame.shape,  senxor_temperature_map_m08.shape, MLX_temperature_map.shape,)
        interm1 = np.concatenate((realsense_depth_image, realsense_color_image, senxor_temperature_map_m08), axis=1)
        # black image: shape is 320*2 by 240
        
        put_text(mask, "human mask")
        
        
        
        # ================================== arrange the images for visualization ==================================
        if inference_mode == 1:
            interm2 = np.concatenate((seek_camera_frame, mask, senxor_temperature_map_m16), axis=1)
            interm1 = np.concatenate((interm1, interm2), axis=1)
            interm1 = np.concatenate((interm1, point_cloud_image), axis=0)
            final_image = interm1
        elif inference_mode == 0: 
            interm2 = np.concatenate((seek_camera_frame, mask, senxor_temperature_map_m16), axis=1)
            interm1 = np.concatenate((interm1, interm2), axis=0)
            interm1 = np.concatenate((interm1, point_cloud_image), axis=0)
            final_image = interm1
        else:
            black_image = np.zeros((240, 320, 3), dtype=np.uint8)

            interm2 = np.concatenate((seek_camera_frame, black_image, senxor_temperature_map_m16), axis=1)
            interm1 = np.concatenate((interm1, interm2), axis=0)
            final_image = interm1
        return final_image
            
        
    def prepare_one_visual(self, realsense_color_image, realsense_depth_image, thermal, point_cloud_image):
        realsense_depth_image = cv2.applyColorMap(cv2.convertScaleAbs(realsense_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        realsense_depth_image = cv2.resize(realsense_depth_image, (320, 240))
        realsense_color_image = cv2.resize(realsense_color_image, (320, 240), interpolation=cv2.INTER_NEAREST)
        
        # visualize m08
        thermal_min = -1024
        thermal_max = -1024
        thermal_min = np.min(thermal)
        thermal_max = np.max(thermal)
        senxor_temperature_map_thermal = thermal.astype(np.uint8)
        senxor_temperature_map_thermal = cv2.normalize(senxor_temperature_map_thermal, None, 0, 255, cv2.NORM_MINMAX)
        senxor_temperature_map_thermal = cv2.resize(senxor_temperature_map_thermal, (320, 240), interpolation=cv2.INTER_NEAREST)
        senxor_temperature_map_thermal = cv2.applyColorMap(senxor_temperature_map_thermal, cv2.COLORMAP_JET)
        put_temp(senxor_temperature_map_thermal, thermal_min, thermal_max, "thermal")
        put_text(realsense_color_image, "color")
        put_text(realsense_depth_image, "depth")

        interm2 = np.concatenate((realsense_color_image, realsense_depth_image, senxor_temperature_map_thermal), axis=1)
        print(interm2.shape, point_cloud_image.shape)
        interm1 = np.concatenate((interm2, point_cloud_image), axis=0)
        return interm1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_duration", type=int, default=60, help="Duration to collect data, seconds")
    parser.add_argument("--sleep_time", type=float, default=0, help="sleep time between each frame")
    parser.add_argument("--enable_MLX", type=int, default=1, help="enable MLX or not")
    parser.add_argument("--mi08_process", type=int, default=0, help="enable postprocessing for mi08 or not")
    parser.add_argument("--mi16_process", type=int, default=0, help="enable postprocessing for mi16 or not")
    parser.add_argument("--save", type=int, default=0, help="0 for not save, 1 for save")
    timestampstr = time.strftime("%Y%m%d%H%M%S", time.localtime()) + f"{int((time.time()%1)*1e6):06d}"
    parser.add_argument("--save_dest", type=str, default=f"data/{timestampstr}", help="destination for saving image, thermal and depth maps")

    parser.add_argument("--exp_config_file", type=str, help="Configuration YAML file of the experiment")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth weights (optional)")
    parser.add_argument("--train", type=int, default="0", help="0 is test, 1 is train")
    parser.add_argument("--thermal_input", type=str, default="m08", help="choose from m08, m16 and seek")
    parser.add_argument("--inference", type=int, default=1, help="whether to run inference or not, 1 for inference, 0 for no inference, -1 for no annotation and no inference")
    parser.add_argument("--use_old_plot", type=int, default=0, help="whether to use old plot or not")
    parser.add_argument("--sensor_type", type=str, default="m08", help="choose from m08, m16 and seek")
    args = parser.parse_args()
    
    exp_config_file_name = args.exp_config_file + '.yaml'
    exp_config_file_name_full = "exp_configs/" + args.exp_config_file + '.yaml'
    exp_config = yaml.safe_load(open(exp_config_file_name_full))
    
    t2p = M08ToPtcloud('exp_configs', exp_config_file_name, args.weights)

    imgdest = os.path.join(args.save_dest, "realsense_color")
    depthdest = os.path.join(args.save_dest, "realsense_depth")
    thermal_dest = os.path.join(args.save_dest, args.thermal_input)
    pointcloudoutputdest = os.path.join(args.save_dest, "pointcloud_output")
    annotationdest = os.path.join(args.save_dest, "annotation")
    
    thermal_input = args.thermal_input
    sensor_name = "seek_thermal" if thermal_input == "seek" else f"senxor_{thermal_input}"
    annotator = DataAnnotate(sensor_name)
    
    if args.save == 1 and not os.path.exists(args.save_dest):
        os.mkdir(args.save_dest)
        os.mkdir(imgdest)
        os.mkdir(depthdest)
        os.mkdir(thermal_dest)
        os.mkdir(pointcloudoutputdest)
        os.mkdir(annotationdest)

    if args.mi08_process or args.mi16_process:
        senxor_postprocess_m = senxor_postprocess()

    realsense_sensor = realsense()
    if args.sensor_type == "m08" or args.sensor_type == "m16":
        senxor_sensor = senxor_16(sensor_port="/dev/ttyACM1") #beware! This may get flipped
    num_rows_senxor, num_cols_senxor = senxor_sensor.get_temperature_map_shape()
    # if num_rows_senxor != 62 or num_cols_senxor != 80:
    #     senxor_sensor = senxor_16(sensor_port="/dev/ttyACM1") #beware! This may get flipped

    # seek
    if args.sensor_type == "seek":
        seek_sensor = seekthermal(data_format="others")

    # buffer for synchronizing different sensors
    # since some sensors get data slower
    buffer_len = 3
    seek_camera_buffer = image_buffer(buffer_len)
    realsense_color_buffer = image_buffer(buffer_len)
    realsense_depth_buffer = image_buffer(buffer_len)

    # prepare shapes of the inputs
    senxor_sensor_shape = senxor_sensor.get_temperature_map_shape()

    # metadata about collection timing
    framecnt = 0   # the number of the received frames
    saved_frame_cnt = 0  # the number of the saved frames
    start_time = time.time()
    collection_duration = args.collection_duration
    sleep_time = args.sleep_time   # sleep time between each frame, control the collecting speed
    last_collect_time = time.time()

    # preparation for plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
        
    # plt.show(block=False)






    dataProcessor = DataProcessor()

        
    while True:
        #print("===========debug: start collecting data, frame:", framecnt, "================")
        framecnt+=1
        
        # obtain data from sensors
        
        
        if args.sensor_type == "m08" or args.sensor_type == "m16":
            temp_ori, header1 = senxor_sensor.get_temperature_map()
            temp_ori = temp_ori.reshape(num_cols_senxor, num_rows_senxor)
            temp_ori = np.flip(temp_ori, 0)
            
            print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDD:", temp_ori.shape)

        realsense_depth_image_ori, realsense_color_image_ori = realsense_sensor.get_frame()
            
        if args.sensor_type == "seek":
            seek_camera_frame_ori = copy.deepcopy(seek_sensor.get_frame())
            seek_camera_buffer.add(seek_camera_frame_ori)
            temp_ori= seek_camera_buffer.get()
            temp_ori = np.flip(temp_ori, 0)
            temp_ori = np.flip(temp_ori, 1)

        # adding to buffer for synchronization
        realsense_color_buffer.add(realsense_color_image_ori)
        realsense_depth_buffer.add(realsense_depth_image_ori)
        
        # drawing from buffer for synchronization
        realsense_color_image_ori = realsense_color_buffer.get()
        realsense_depth_image_ori = realsense_depth_buffer.get()
        
        
        
        

        # ================================== check exist and processing data ==================================
        if realsense_depth_image_ori is None or realsense_color_image_ori is None or temp_ori is None:
            continue
        else:
            
            
            # ================================== saving the raw data ==================================
            # preparing file name
            timestampstr = time.strftime("%Y%m%d%H%M%S", time.localtime()) + f"{int((time.time()%1)*1e6):06d}"
            npyname = timestampstr + ".npy"
            pklname = timestampstr + ".pkl"
            if args.save == 1:
                dataProcessor.save_raw_data(realsense_depth_image_ori, realsense_color_image_ori, thermal_dest, timestampstr)









            # ================================== Inference: get the predicted point clouds ================================================
            if args.inference == 1:
                ptcloud = dataProcessor.get_point_clouds_pred(t2p, temp_ori)
                if args.save == 1:
                    dataProcessor.save_pcd_pred(ptcloud, timestampstr, pointcloudoutputdest)
            # if args.inference == 1 or args.inference == 0:
            #     result_dict = dataProcessor.get_annotation(realsense_color_image_ori, realsense_depth_image_ori, annotator)
            #     if args.save == 1:
            #         dataProcessor.save_annotation(result_dict, timestampstr, annotationdest)

            timestamp = time.time()
            
            
            
            
            

            # ================== for visualization of point clouds: 2 axes for inference and annotate, 1 axis for annotate, no axis for collection ============
            # the order should not be changed because we need to plot on two axes and obtain final image.
            pcd_image = None
            # if args.inference == 1 or args.inference == 0:
                # pcd_image = dataProcessor.visualize_gt_pcd(fig, ax, result_dict, use_old_plot=args.use_old_plot, no_id_distinguish=True)
            if args.inference == 1:
                print(ptcloud.shape, "DDDEEEBBBUUUGGG~~~~~~~~~~~~~~~~")
                print("~~~~~~~use old plot:", args.use_old_plot)
                if args.use_old_plot:
                    pcd_image = dataProcessor.visualize_pred_pcd(fig, ax, ptcloud, exp_config, use_old_plot=True, no_id_distinguish=False)
                else:
                    # print("DEBUG: shape of idx0:", pcd_image.shape)
                    pcd_image = dataProcessor.visualize_pred_pcd(fig, ax, ptcloud, exp_config, use_old_plot=False, no_id_distinguish=False)

            # # visualize mask
            mask = None
            # if args.inference != -1:
            #     mask = process_mask(result_dict)
                
                
                
                
                
            # ================================== Prepare the images for visualization ==================================
            # visualize realsense
            final_image = dataProcessor.prepare_one_visual(realsense_color_image_ori, realsense_depth_image_ori, temp_ori, pcd_image)
            cv2.imshow("Sensor Visuals", final_image)


            
            
            # ================================== check if we overrun ==================================
            # time_lasting = time.time() - start_time
            # if time_lasting > collection_duration:
            #     break
            #     # timestamp = time.time()
            #     # print(f"Realsense depth and color image collected at {timestamp}", realsense_depth_image.shape, realsense_color_image.shape)
            #     # print(f"Senxor temperature map m08 collected at {timestamp}", senxor_temperature_map_m08.shape)
            # print(f"Total frames received: {framecnt}")
            # print(f"Frame rate: {framecnt / time_lasting} Hz")

                #break

            key = cv.waitKey(1)
            if key in [ord("q"), ord('Q'), 27]:
                break

    # senxor_sensor_m08.close()

    for i in range (5):
        print('\a')

# python data_collection.py --save_data 0 --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth