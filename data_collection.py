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
import matplotlib.pyplot as plt
import logging
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
    def __init__(self, sensor_port = "/dev/ttyACM1"):
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

def plot_3d_point_cloud(fig, ax, point_cloud, max_num_persons, max_num_points, camera_height=1, elev=15, azim=-45, threshold=0.5, s= 10):
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
    # colors = plt.cm.jet(np.linspace(0, 1, max_num_persons))
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
                        label="", alpha=0.5, s=s, c="red")
    ax.set_xlim([-2000, 2000])
    ax.set_ylim([0, 4000])
    ax.set_zlim([-1000, 1000])
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_duration", type=int, default=60, help="Duration to collect data, seconds")
    parser.add_argument("--sleep_time", type=float, default=0, help="sleep time between each frame")
    parser.add_argument("--enable_MLX", type=int, default=1, help="enable MLX or not")
    parser.add_argument("--mi08_process", type=int, default=0, help="enable postprocessing for mi08 or not")
    parser.add_argument("--mi16_process", type=int, default=0, help="enable postprocessing for mi16 or not")

    parser.add_argument("--exp_config_file", type=str, help="Configuration YAML file of the experiment")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth weights (optional)")
    parser.add_argument("--train", type=int, default="0", help="0 is test, 1 is train")
    args = parser.parse_args()
    exp_config_file_name = args.exp_config_file + '.yaml'
    t2p = M08ToPtcloud('exp_configs', exp_config_file_name, args.weights)
    
    
    if args.mi08_process:
        senxor_postprocess_m08 = senxor_postprocess()

    realsense_sensor = realsense()  
    senxor_sensor_m08 = senxor_16(sensor_port="/dev/ttyACM0") #beware! This may get flipped

    # buffer for synchronizing different sensors
    # since some sensors get data slower
    buffer_len = 3

    # seek_camera_buffer = image_buffer(buffer_len)
    realsense_color_buffer = image_buffer(buffer_len)
    realsense_depth_buffer = image_buffer(buffer_len)
    # mlx_buffer = image_buffer(buffer_len)
    
    num_rows_m08, num_cols_m08 = senxor_sensor_m08.get_temperature_map_shape()
    # num_rows_m16, num_cols_m16 = senxor_sensor_m16.get_temperature_map_shape()

    print("before collecting data=================================================")
    framecnt = 0   # the number of the received frames
    saved_frame_cnt = 0  # the number of the saved frames
    start_time = time.time()
    collection_duration = args.collection_duration
    sleep_time = args.sleep_time   # sleep time between each frame, control the collecting speed
    last_collect_time = time.time()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # plt.show(block=False)

    while True:
        #print("===========debug: start collecting data, frame:", framecnt, "================") 
        framecnt+=1
        senxor_temperature_map_m08_ori, header1 = senxor_sensor_m08.get_temperature_map()
        # senxor_temperature_map_m16_ori, header2 = senxor_sensor_m16.get_temperature_map()
        realsense_depth_image_ori, realsense_color_image_ori = realsense_sensor.get_frame()
        # seek_camera_frame_ori = copy.deepcopy(seek_camera.get_frame())
        # if args.enable_MLX:
        #     MLX_temperature_map_ori = mlx_sensor.get_temperature_map()
        #     mlx_buffer.add(MLX_temperature_map_ori)
        #     MLX_temperature_map_ori = mlx_buffer.get()
        
        # seek_camera_buffer.add(seek_camera_frame_ori)
        realsense_color_buffer.add(realsense_color_image_ori)
        realsense_depth_buffer.add(realsense_depth_image_ori)           

        # seek_camera_frame_ori= seek_camera_buffer.get()
        realsense_color_image_ori = realsense_color_buffer.get()
        realsense_depth_image_ori = realsense_depth_buffer.get()

        if realsense_depth_image_ori is None or realsense_color_image_ori is None or senxor_temperature_map_m08_ori is None:
            continue
        else:
            realsense_depth_image, realsense_color_image, senxor_temperature_map_m08 = realsense_depth_image_ori, realsense_color_image_ori, senxor_temperature_map_m08_ori

            
            senxor_temperature_map_m08 = senxor_temperature_map_m08.reshape(num_cols_m08, num_rows_m08)
            senxor_temperature_map_m08 = np.flip(senxor_temperature_map_m08, 0)
            if args.mi08_process:
                senxor_temperature_map_m08 = senxor_postprocess_m08.process_temperature_map(senxor_temperature_map_m08)
            
            print("shape of m08:", senxor_temperature_map_m08.shape)
            thermal_images = np.expand_dims(senxor_temperature_map_m08, axis=0)
            thermal_images = np.expand_dims(thermal_images, axis=0)
            print("shape of m08 afterwards:", thermal_images.shape)
            thermal_images = torch.from_numpy(thermal_images.copy())
            # produce point cloud visualization for m08
            ptcloud = t2p.thermal2ptcloud(thermal_images)

            timestamp = time.time()
            
            # for visualization only
            # if args.vis_flag:
            # visualize point cloud
            ax.clear()
            plot_3d_point_cloud(fig, ax, ptcloud.cpu().numpy(), 6, 1000)
            fig.canvas.draw()
            fig.canvas.flush_events()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # plt.show()
            # rescale image such that its width is 960, and its height-width ration remains unchanged
            image = cv2.resize(image, (960, int(960 * image.shape[0] / image.shape[1])))
            
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
            #print(realsense_depth_image.shape, realsense_color_image.shape, seek_camera_frame.shape,  senxor_temperature_map_m08.shape, MLX_temperature_map.shape,)
            interm1 = np.concatenate((realsense_depth_image, realsense_color_image, senxor_temperature_map_m08), axis=1)
            interm1 = np.concatenate((interm1, image), axis=0)
            final_image = interm1
            cv2.imshow("Final Image", final_image)
                
            time_lasting = time.time() - start_time
            if time_lasting > collection_duration:
                break
                # timestamp = time.time()
                # print(f"Realsense depth and color image collected at {timestamp}", realsense_depth_image.shape, realsense_color_image.shape)
                # print(f"Senxor temperature map m08 collected at {timestamp}", senxor_temperature_map_m08.shape)
            print(f"Total frames received: {framecnt}")
            print(f"Frame rate: {framecnt / time_lasting} Hz")

                #break
            
            key = cv.waitKey(1)
            if key in [ord("q"), ord('Q'), 27]:
                break
        
    senxor_sensor_m08.close()

    for i in range (5):
        print('\a')
        
# python data_collection.py --save_data 0 --exp_config_file model3_m08 --weights weights/m08/model3_m08_thermo_pt_0819203728.pth