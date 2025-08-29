import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from ImageAligner import ImageAligner
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo # model zoo: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
from utils import Get_COCO_pose_joint_label, colorize_thermal_map, calculate_iou, raw_data_loader
from tqdm import tqdm
import pickle
import time
import argparse

class DataAnnotate():
    def __init__(self, thermal_sensor_name):
        self.thermal_sensor_name = thermal_sensor_name

        # loading the pretrained model for labeling
        seg_cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        seg_cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'))  #  "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        seg_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        seg_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml')  
        # model link: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn
        self.segmentation_predictor = DefaultPredictor(seg_cfg)

        # Inference with a keypoint detection model
        pose_cfg = get_cfg()   # get a fresh new config
        pose_cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))  # COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml
        pose_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        pose_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
        self.keypoints_predictor = DefaultPredictor(pose_cfg)

        # image aligner
        if thermal_sensor_name == 'seek_thermal':
            data_shape = (150,200)
        elif thermal_sensor_name == 'senxor_m08':
            data_shape = (62, 80)
        elif thermal_sensor_name == 'senxor_m16':
            data_shape = (120, 160)
        else:
            data_shape = None
            raise ValueError('The thermal sensor name is not supported')
        self.it = ImageAligner(thermal_sensor_name, data_shape)

    def forward(self, color, depth):
        results_dict = {
            'num_persons': 0,
            'depth_person': [],
            'depth_mask_person': [],  
            'point_cloud_person': [],
            '2D_pose_person': [],
        }
        
        segmentation_outputs = self.segmentation_predictor(color)
        seg_result = segmentation_outputs["instances"].to("cpu")
        seg_classes = seg_result.pred_classes.numpy()
        human_index = np.where(seg_classes == 0)
        seg_result_person = seg_result[human_index]
        # human related segmentation results
        seg_masks = seg_result_person.pred_masks.numpy()
        # seg_scores = seg_result_person.scores.numpy()  
        seg_boxes = seg_result_person.pred_boxes.tensor.numpy() 
        # print(seg_boxes)

        keypoints_outputs = self.keypoints_predictor(color)
        keypoints_result = keypoints_outputs["instances"].to("cpu")
        keypoints_classes = keypoints_result.pred_classes.numpy()
        human_index = np.where(keypoints_classes == 0)
        # human related keypoints results
        keypoints_result_person = keypoints_result[human_index]
        # keypoints_scores = keypoints_result_person.scores.numpy()
        keypoints_keypoints = keypoints_result_person.pred_keypoints.numpy()
        keypoints_boxes = keypoints_result_person.pred_boxes.tensor.numpy()
        # print(keypoints_boxes)

        # identify the person in both segmentation and keypoints
        iou = np.zeros((len(seg_boxes), len(keypoints_boxes)))
        for i in range(len(seg_boxes)):
            for j in range(len(keypoints_boxes)):
                iou[i, j] = calculate_iou(seg_boxes[i], keypoints_boxes[j])
        # print(iou)

        for i in range(len(seg_boxes)):  # traverse the segmentation results
            try:
                max_iou = np.max(iou[i, :])
                max_iou_index = np.argmax(iou[i, :])
            except:
                continue
            if max_iou > 0.8:
                mask = seg_masks[i]             # the mask of the person
                person_depth = depth*mask       # the depth of the person, now the x,y is the pixel coordinate
                # person_depth_save = person_depth.copy()  # for later saving
                point_cloud_map = self.depth_map_to_point_cloud(person_depth, hfov_deg = 88.5, vfov_deg = 58.5)  # the point cloud map (x_cam,y_cam,depth) of the person, 
                                                                                                            # now the x,y is the camera coordinate (unit: mm)
                # filtering out the outliers based on the depth
                person_non_zero = person_depth[person_depth > 30]  # filtering out the points with depth less than 30 mm
                # print(len(person_non_zero)) 
                # Aligning the depth map to thermal map
                try:
                    representative_value = self.find_representative_value(person_non_zero.flatten())    # the representative value of the depth map, i.e., the depth of the person
                except:
                    continue
                aligner_parameter = self.depth_aligner_parameter_mapping(representative_value)    
                if aligner_parameter is None:
                    continue 
                try:
                    point_cloud_map = self.it.transform_image(point_cloud_map, self.it.load_transformation(aligner_parameter))    
                except:
                    continue
                
                point_cloud_list = []
                # calculating the outlier removal bound
                lower_bound, upper_bound = self.outlier_removal_bound(person_non_zero.flatten())
                # print(lower_bound, upper_bound)
                for i in range(point_cloud_map.shape[0]):
                    for j in range(point_cloud_map.shape[1]):
                        if point_cloud_map[i, j,2] > 30 and point_cloud_map[i, j,2] > lower_bound and point_cloud_map[i, j,2] < upper_bound:
                            point_cloud_list.append(point_cloud_map[i, j])
                pt_cloud = np.array(point_cloud_list) # the point cloud of the person
                # print('pt_cloud:', pt_cloud.shape)
                
                # get the 2D pose and converting to the thermal camera's view
                pose_2D = keypoints_keypoints[max_iou_index]  # pose shape: (17, 3) 17 keypoints with (x, y, visibility)
                transformed_2D_pose = np.zeros((pose_2D.shape[0], 3))
                for index,keypoint in enumerate(pose_2D):
                    # generate a pose map with shape of the color image, and the value is the index of the keypoint
                    pose_map = np.zeros((color.shape[0], color.shape[1], 3))
                    keypoint = keypoint.astype(np.int32)
                    # pose_map[keypoint[1], keypoint[0], 0] = 255 
                    # the near neighbor with the value of 255
                    pose_map[keypoint[1]-4:keypoint[1]+4, keypoint[0]-4:keypoint[0]+4, 0] = 255
                    pose_map = self.it.transform_image(pose_map, self.it.load_transformation(aligner_parameter)) 
                    # find the index of the element with the value over 50
                    indexs = np.where(pose_map[...,0] > 0.001)
                    # calculating the average of the indexs
                    x = np.mean(indexs[1])
                    y = np.mean(indexs[0])
                    transformed_2D_pose[index] = [x, y, pose_2D[index, 2]]
                
                results_dict['num_persons'] += 1
                results_dict['depth_person'].append(representative_value)
                results_dict['depth_mask_person'].append(point_cloud_map[:,:,2])
                results_dict['point_cloud_person'].append(pt_cloud)
                results_dict['2D_pose_person'].append(transformed_2D_pose)
        return results_dict
                

    def find_representative_value(self,arr):
        '''
            Find the representative value of the array
        '''
        # Step 1: Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = np.percentile(arr, 25)
        Q3 = np.percentile(arr, 75)
        # print('Q1:', Q1, 'Q3:', Q3)
        # Step 2: Calculate the interquartile range (IQR)
        IQR = Q3 - Q1
        # print('IQR:', IQR)
        # Step 3: Define the lower and upper bounds to identify outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # print('lower_bound:', lower_bound, 'upper_bound:', upper_bound)
        # Step 4: Filter out the outliers
        filtered_arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]
        # print('filtered_arr:', filtered_arr)
        # Step 5: Compute the center of the remaining values (e.g., median or mean)
        representative_value = np.median(filtered_arr)  # or np.mean(filtered_arr)
        # print('representative_value:', representative_value)
        return representative_value

    def outlier_removal_bound(self, arr):
        """
        Find the lower and upper bound for the outlier removal
        """
        # Step 1: Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = np.percentile(arr, 25)
        Q3 = np.percentile(arr, 75)
        # print('Q1:', Q1, 'Q3:', Q3)
        # Step 2: Calculate the interquartile range (IQR)
        IQR = Q3 - Q1
        # print('IQR:', IQR)
        # Step 3: Define the lower and upper bounds to identify outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # print('lower_bound:', lower_bound, 'upper_bound:', upper_bound)
        # Step 4: Filter out the outliers
        # filtered_arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]
        return lower_bound, upper_bound 

    def depth_aligner_parameter_mapping(self, depth):
        """
        Mapping the depth value to the depth aligner parameter
        
        Parameters
        ----------
        depth : float
            The depth value
        
        Returns
        -------
        int
            The depth aligner parameter
        """
        if depth < 500:
            return "250-500"
        elif depth < 750:
            return "500-750"
        elif depth < 1000:
            return "750-1000"
        elif depth < 1250:
            return "1000-1250"
        elif depth < 1500:
            return "1250-1500"
        elif depth < 2000:
            return "1500-2000"
        elif depth < 2500:
            return "2000-2500"
        elif depth < 3000:
            return "2500-3000"
        elif depth < 3500:
            return "3000-3500"
        elif depth < 4000:
            return "3500-4000"
        elif depth < 4500:
            return "4000-4500"
        elif depth < 5000:
            return "4500-5000"
        elif depth < 5500:
            return "5000-5500"
        elif depth < 6000:
            return "5500-6000"
        elif depth < 6500:
            return "6000-6500"
        elif depth < 7000:
            return "6500-7000"
        elif depth < 7500:
            return "7000-7500"
        elif depth < 8000:
            return "7500-8000"
        else:
            return None

    def depth_map_to_point_cloud(self, depth_map, hfov_deg, vfov_deg):
        """
        Convert a depth map to a 3D point cloud.

        Parameters:
        depth_map: 2D numpy array, the depth map (in meters or any consistent unit).
        hfov_deg: float, horizontal field of view of the camera in degrees.
        vfov_deg: float, vertical field of view of the camera in degrees.

        Returns:
        point_cloud: 3D numpy array of shape (H, W, 3), where (H, W) is the depth map shape.
                    Each point has (x, y, z) coordinates in 3D space.
        """
        # Get depth map dimensions
        height, width = depth_map.shape

        # Convert HFoV and VFoV from degrees to radians
        hfov_rad = np.deg2rad(hfov_deg)
        vfov_rad = np.deg2rad(vfov_deg)

        # Calculate focal lengths (fx, fy) based on FoV and image dimensions
        fx = width / (2 * np.tan(hfov_rad / 2))
        fy = height / (2 * np.tan(vfov_rad / 2))
        # Create a grid of pixel coordinates
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        # Normalize pixel coordinates to camera coordinates
        x_cam = (x - width / 2) / fx
        y_cam = (y - height / 2) / fy
        # Calculate 3D coordinates
        z_cam = depth_map
        x_3d = x_cam * z_cam
        y_3d = y_cam * z_cam

        # Stack the coordinates to form the point cloud
        point_cloud = np.stack((x_3d, y_3d, z_cam), axis=-1)
        return point_cloud


if __name__ == "__main__":
    # use command:
    # python DataAnnotation.py --sensor_name senxor_m16 --visualization_flag 1 --raw_data_folder RawData2
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_name', type=str, default='senxor_m16', help='the name of the sensor')
    parser.add_argument('--visualization_flag', type=int, default=1, help='whether to save a video for visualization of the annotation results')
    parser.add_argument('--raw_data_folder', type=str, default='RawData2', help='the folder name of the raw data')
    args = parser.parse_args()
    
    # Log the run configuration at the beginning
    log_file = 'annotation_log.txt'
    with open(log_file, 'a') as f:
        f.write(f"\n=== Annotation Run: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Configuration:\n")
        f.write(f"- Sensor name: {args.sensor_name}\n")
        f.write(f"- Raw data folder: {args.raw_data_folder}\n")
        f.write(f"- Visualization: {'Enabled' if args.visualization_flag == 1 else 'Disabled'}\n")
        f.write(f"Starting annotation process...\n\n")
    
    sensor_name = args.sensor_name   # 'seek_thermal', 'senxor_m08', 'senxor_m16'  
    if args.visualization_flag == 1:    
        visualization_flag = True
    else:
        visualization_flag = False
    raw_data_folder = args.raw_data_folder
    selected_recording_folder = []  # means all the folders in the raw data folder  'U0o1_E2_2_walking_1o3_none_0', 'U0o1o2_E2_3_walking_1o3_none_1'
    
    annotator = DataAnnotate(sensor_name)
    slected_index_list = []   # means all the samples
    
    ## the data saving folder
    save_folder = 'AnnotatedData'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    if len(selected_recording_folder) == 0:
        selected_recording_folder = os.listdir(raw_data_folder)
    
    if visualization_flag:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        time_str = time.strftime("%Y%m%d-%H%M%S")
        video_name = raw_data_folder + '_' + sensor_name + '_' + time_str + '.avi'
        out = cv2.VideoWriter(os.path.join(save_folder, video_name), fourcc, 50.0, (1920, 480))
    
    failed_folder_list = []
    existed_folder_list = []
    for folder_index, recording_folder in enumerate(selected_recording_folder):
        try:
            print('starting to process:')
            print('folder_index:', folder_index, 'total number of folders:', len(selected_recording_folder), 'folder_name:', recording_folder)
            # making the folder for saving the annotated data with the same structure as the raw data folder
            if not os.path.exists(os.path.join(save_folder, recording_folder)):
                os.makedirs(os.path.join(save_folder, recording_folder))
            # making a subfolder with the sensor name
            if not os.path.exists(os.path.join(save_folder, recording_folder, sensor_name)):
                os.makedirs(os.path.join(save_folder, recording_folder, sensor_name))
            else:  # the folder exists, which means the data has been annotated or their is a duplicate in the name
                print('The folder exists:', os.path.join(save_folder, recording_folder, sensor_name))
                # Check if the folder is empty
                raw_data_length = len(os.listdir(os.path.join(raw_data_folder, recording_folder, sensor_name)))
                if raw_data_length > 100:
                    if len(os.listdir(os.path.join(save_folder, recording_folder, sensor_name))) < raw_data_length-2:
                        print('The data is not complete, proceeding with annotation...')
                    else:
                        print('The data is complete, skipping annotation...')
                        existed_folder_list.append(recording_folder)
                        continue
                else:
                    if len(os.listdir(os.path.join(save_folder, recording_folder, sensor_name))) < 50: # each recording has more than 50 samples
                        print('The folder is empty, proceeding with annotation...')
                    else:
                        print('The folder contains data, skipping annotation...')
                        existed_folder_list.append(recording_folder)
                        continue
                    
            # loading the  raw data
            print('loading the raw data...')
            seek_camera_data, MLX_data, senxor_m08_data, senxor_m16_data, realsense_depth_data, realsense_color_data, file_name_list = raw_data_loader(raw_data_folder, 
                                                                                                                                                    recording_folder, 
                                                                                                                                                    slected_index_list, 
                                                                                                                                                    file_name_list_flag=True)
            print('raw data loaded, start to annotate...')
            
            if len(file_name_list) != len(seek_camera_data) or len(file_name_list) != len(MLX_data) or len(file_name_list) != len(senxor_m08_data) or len(file_name_list) != len(senxor_m16_data) or len(file_name_list) != len(realsense_depth_data) or len(file_name_list) != len(realsense_color_data):
                raise ValueError('The number of the samples is not consistent')
            
            annotation = []
            count = 0   
            for sample_index in tqdm(range(len(seek_camera_data))):
                depth = realsense_depth_data[sample_index]
                color = realsense_color_data[sample_index]  
                if sensor_name == 'seek_thermal':
                    thermal = seek_camera_data[sample_index]
                elif sensor_name == 'senxor_m08':
                    thermal = senxor_m08_data[sample_index]
                elif sensor_name == 'senxor_m16':
                    thermal = senxor_m16_data[sample_index]
                else:
                    raise ValueError('The thermal sensor name is not supported')
                if thermal is None or depth is None or color is None:
                    continue
                results_dict = annotator.forward(color, depth)
                file_name = file_name_list[sample_index][:-4]  # remove the '.npy' suffix
                # save the results as pickle use the file name and also the thermal data as numpy array
                pickle.dump(results_dict, open(os.path.join(save_folder, recording_folder, sensor_name, file_name + '.pkl'), 'wb'))
                np.save(os.path.join(save_folder, recording_folder, sensor_name, file_name + '.npy'), thermal)
                count += 1
                
                if visualization_flag:
                    thermal_colorized = colorize_thermal_map(thermal)
                    thermal_colorized = thermal.astype(np.uint8)
                    thermal_colorized = cv2.normalize(thermal_colorized, None, 0, 255, cv2.NORM_MINMAX)
                    thermal_colorized = cv2.applyColorMap(thermal_colorized, cv2.COLORMAP_JET)
                    # print('thermal_colorized:', thermal_colorized.shape)
                    depth_masks = np.zeros((thermal.shape[0], thermal.shape[1]))
                    
                    for select_person_index in range(results_dict['num_persons']):
                        depth_mask = results_dict['depth_mask_person'][select_person_index]
                        depth_masks += depth_mask
                        pose = results_dict['2D_pose_person'][select_person_index]
                        for index,keypoint in enumerate(pose):
                            # Check for invalid values
                            invalid_kp_index = np.isnan(keypoint) | np.isinf(keypoint)
                            # Handle invalid values (replace with 0 in this case)
                            keypoint[invalid_kp_index] = 0  # Replace NaN and inf with 0
                            keypoint = keypoint.astype(np.int32)
                            cv2.circle(thermal_colorized, (keypoint[0], keypoint[1]), 1, (255, 255, 255), -1)
                            # get the label of the keypoint
                            # label = Get_COCO_pose_joint_label(index)
                            # cv2.putText(thermal_colorized, label, (keypoint[0], keypoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                    depth_masks = cv2.applyColorMap(cv2.convertScaleAbs(depth_masks, alpha=0.03), cv2.COLORMAP_JET)
                    # resize both the thermal and depth mask as the size of the color image
                    thermal_colorized = cv2.resize(thermal_colorized, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # convert the rgba thermal colorized image to rgb
                    depth_masks = cv2.resize(depth_masks, (color.shape[1], color.shape[0]))
                    # concatenate the thermal, depth mask and color image
                    vis = np.concatenate((thermal_colorized[:,:,:3], depth_masks, color), axis=1)
                    # put the file name on the image
                    name = os.path.join(raw_data_folder, recording_folder, sensor_name, file_name)
                    cv2.putText(vis, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) 
                    # get dist of human depth mask

                    #cv2.putText(vis, dist, (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    if visualization_flag:
                        out.write(vis)
            print('total number of samples:', count)
            if count == 0:
                failed_folder_list.append(recording_folder)
        except Exception as e:
            # print('error in the folder:', recording_folder, 'error:', e)
            failed_folder_list.append(recording_folder)
    if visualization_flag:
        out.release()
        print('video saved')
    print('failed_folder_list:', failed_folder_list)
    # Write failed folders to the log file
    with open(log_file, 'a') as f:
        f.write(f"Annotation results:\n")
        f.write(f"- Total folders processed: {len(selected_recording_folder)}\n")
        if failed_folder_list:
            f.write(f"- Failed folders ({len(failed_folder_list)}):\n")
            for folder in failed_folder_list:
                f.write(f"  * {folder}\n")
        else:
            f.write(f"- No failed folders\n")
            
        if existed_folder_list:
            f.write(f"- Already processed folders ({len(existed_folder_list)}):\n")
            for folder in existed_folder_list:
                f.write(f"  * {folder}\n")
        f.write(f"=== End of Run ===\n\n")