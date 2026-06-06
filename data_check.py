import serial
import time
import ast
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
from inference_new import M08ToPtcloud, plot_3d_point_cloud
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
matplotlib.use('Agg')  # Ensure Agg backend is set
import matplotlib.pyplot as plt
import logging
from DataAnnotation import DataAnnotate
import pickle as pkl
from plot import plot_3d_point_cloud_new, remove_small_regions, mark_connected_components
import yaml
from image2vid import Img2Vid

logging.getLogger().setLevel(logging.CRITICAL)
# sys.path.append("/home/zx/Desktop/zx/DeepTadarDataCollect-ubuntu-data-collect/")
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

def plot_3d_point_cloud(fig, ax, point_cloud, max_num_persons, max_num_points, camera_height=1, elev=15, azim=-45, threshold=0.1, s= 10):
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


    plot_camera(ax)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    # Plot points for each person
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
    ax.set_xlim([-2000, 2000])
    ax.set_ylim([0, 4000])
    ax.set_zlim([-1000, 1000])
    # how can I change the viewing angle?
    # ans:
    # ax.view_init(elev=20, azim=-0)

from data_collection import DataProcessor, process_mask
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--collection_duration", type=int, default=60, help="Duration to collect data, seconds")
    # parser.add_argument("--sleep_time", type=float, default=0, help="sleep time between each frame")
    # parser.add_argument("--enable_MLX", type=int, default=1, help="enable MLX or not")
    # parser.add_argument("--mi08_process", type=int, default=0, help="enable postprocessing for mi08 or not")
    # parser.add_argument("--mi16_process", type=int, default=0, help="enable postprocessing for mi16 or not")
    # parser.add_argument("--save", type=int, default=0, help="0 for not save, 1 for save")
    # timestampstr = time.strftime("%Y%m%d%H%M%S", time.localtime()) + f"{int((time.time()%1)*1e6):06d}"
    # parser.add_argument("--save_dest", type=str, default=f"data/{timestampstr}", help="destination for saving image, thermal and depth maps")

    parser.add_argument("--exp_config_file", type=str, help="Configuration YAML file of the experiment")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth weights (optional)")
    # parser.add_argument("--train", type=int, default="0", help="0 is test, 1 is train")
    parser.add_argument("--thermal_input", type=str, default="m08", help="choose from m08, m16 and seek")
    parser.add_argument("--path", type=str, default=".", help="Path to the data directory")
    parser.add_argument("--use_old_plot", type=int, default=0, help="whether to use old plot or new plot")
    parser.add_argument("--no_id_distinguish", type=int, default=0, help="whether to distinguish between different persons or not")
    parser.add_argument("--annotation", type=int, default=0, help="whether to do annotation or not")
    parser.add_argument("--img2vid", type=int, default=0, help="whether to convert images to video or not")
    parser.add_argument("--vis_mode", type=int, default=-1, help="inference mode (-1: no pred/annotate, 0: annotation, 1: annotation + pred)")
    parser.add_argument("--data_processed", type=int, default=0, help="whether the data has been annotated and processed")
    parser.add_argument("--annotate_and_pred", type=int, default=0, help="whether to do annotation and prediction or not")


    args = parser.parse_args()
    exp_config_file_name = args.exp_config_file + '.yaml'
    # exp_config = yaml.safe_load(open(exp_config_file_name))
    exp_config_file_name_full = "exp_configs/" + args.exp_config_file + '.yaml'

    imgdest = os.path.join(args.path, "realsense_color")
    depthdest = os.path.join(args.path, "realsense_depth")
    m08dest = os.path.join(args.path, "senxor_m08")
    m16dest = os.path.join(args.path, "senxor_m16")
    seekdest = os.path.join(args.path, "seek_color")
    pointcloud_folder_name = "pointcloud_" + args.thermal_input
    pointcloudoutputdest = os.path.join(args.path, pointcloud_folder_name)
    if not os.path.exists(pointcloudoutputdest):
        os.makedirs(pointcloudoutputdest)
    annotationdest = os.path.join(args.path, "annotation")


    # preparation for plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # plt.show(block=False)

    imgpaths = os.listdir(imgdest)
    depthpaths = os.listdir(depthdest)
    m08paths = os.listdir(m08dest)
    m16paths = os.listdir(m16dest)
    seekpaths = os.listdir(seekdest)
    pointcloudpaths = os.listdir(pointcloudoutputdest)
    annotationpaths = os.listdir(annotationdest)

    imgpaths.sort()
    depthpaths.sort()
    m08paths.sort()
    m16paths.sort()
    seekpaths.sort()
    pointcloudpaths.sort()
    annotationpaths.sort()
    
    # vis_mode and annotation/inference status
    vis_mode_switch = {
        -1: (False, False),
        1: (True, True)
    }
    show_annotation, show_inference = vis_mode_switch[args.vis_mode]
    inferenced, annotated = False, False
    if args.data_processed:
        inferenced, annotated = True, True

    framecnt = -1
    sensor_name = "seek_thermal" if args.thermal_input == "seek" else f"senxor_{args.thermal_input}"

    annotate_and_pred = args.annotate_and_pred
    
    if show_annotation or annotate_and_pred:
        annotator = DataAnnotate(sensor_name)
    if show_inference or annotate_and_pred: 
        t2p = M08ToPtcloud('exp_configs', exp_config_file_name, args.weights)
    
    
    
    
    dataProcessor = DataProcessor()
    
    exp_config = yaml.safe_load(open(exp_config_file_name_full))
    
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')

    if args.img2vid:
        img2vid = Img2Vid(args.path + "/output_video.mp4")

    while True:
        print("===============DEBUG: no_id_distinguish:", args.no_id_distinguish)
        #print("===========debug: start collecting data, frame:", framecnt, "================")
        framecnt += 1
        # load all the data from path, one by one
        realsense_depth_image = np.load(os.path.join(depthdest, depthpaths[framecnt]))
        realsense_color_image = np.load(os.path.join(imgdest, imgpaths[framecnt]))
        senxor_temperature_map_m08 = np.load(os.path.join(m08dest, m08paths[framecnt]))
        senxor_temperature_map_m16 = np.load(os.path.join(m16dest, m16paths[framecnt]))
        seek_camera_frame = np.load(os.path.join(seekdest, seekpaths[framecnt]))
        # pointcloud = np.load(os.path.join(pointcloudoutputdest, pointcloudpaths[framecnt]))
        
        timestampstr = imgpaths[framecnt].split("/")[-1].split(".")[0]
        
        # load pickled annotation dictionary
        # we want to visualize the pcd
        if (not annotated and show_annotation) or annotate_and_pred: # annotate & visualize
            result_dict = dataProcessor.get_annotation(realsense_color_image, realsense_depth_image, annotator)
            dataProcessor.save_annotation(result_dict, timestampstr, annotationdest)
        elif annotated: # already annotated, only visualize
            result_dict = pkl.load(open(os.path.join(annotationdest, annotationpaths[framecnt]), "rb"))
        # otherwise, we don't visualize pcd

        # produce point cloud visualization for m08
        # case 1: we want to visualize the pcd
        if (not inferenced and show_inference) or annotate_and_pred: # we need to predict
            ptcloud = dataProcessor.get_point_clouds_pred(args.thermal_input, t2p, senxor_temperature_map_m08, senxor_temperature_map_m16, seek_camera_frame)
            dataProcessor.save_pcd_pred(ptcloud, timestampstr, pointcloudoutputdest)
        elif inferenced: # we already have inference data
            ptcloud = np.load(os.path.join(pointcloudoutputdest, pointcloudpaths[framecnt]))
        # case 2: we don't want to visualize the pcd
        # else, we don't want to inference, and we don't want tosee the inferenced data either. (in the case of checking collected data)
        
        pcd_image = None
        mask = None
        # visualize point cloud
        if args.vis_mode == 1:
            pcd_image = dataProcessor.visualize_gt_pcd(fig, ax, result_dict, use_old_plot=args.use_old_plot, no_id_distinguish=args.no_id_distinguish)
            if args.use_old_plot:
                pcd_image = dataProcessor.visualize_pred_pcd(fig, ax1, ptcloud, exp_config, use_old_plot=True, no_id_distinguish=args.no_id_distinguish)
            else:
                print("DEBUG: shape of idx0:", pcd_image.shape)
                pcd_image = np.concatenate((pcd_image, dataProcessor.visualize_pred_pcd(fig, ax1, ptcloud, exp_config, use_old_plot=False, no_id_distinguish=args.no_id_distinguish)), axis=1)
            #break
            
            mask = process_mask(result_dict)

        final_image = dataProcessor.prepare_sensor_visuals(realsense_color_image, realsense_depth_image, senxor_temperature_map_m08, senxor_temperature_map_m16, seek_camera_frame, pcd_image, mask, args.vis_mode)
        cv2.imshow("Sensor Visuals", final_image)
        print(final_image.shape, "SHAPE OF FINAL IMAGE")

        if args.img2vid:
            img2vid.add_frame(final_image)

        key = cv.waitKey(1)
        if key in [ord("q"), ord('Q'), 27]:
            break

    for i in range (5):
        print('\a')
    if args.img2vid:
        img2vid.release()

# python data_collection.py --save_data 0 --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth