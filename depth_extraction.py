#from DataAnnotation import DataAnnotate, colorize_thermal_map
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import matplotlib.pyplot as plt

import numpy as np
import os

# Set OpenCV to use offscreen backend (for headless servers without display)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2
def calculate_iou(box0, box1):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box0, box1: list or tuple of 4 elements (x1, y1, x2, y2)
                (x1, y1) is the top-left corner,
                (x2, y2) is the bottom-right corner.

    Returns:
    iou: float, the IoU value between the two boxes.
    """
    # Extract coordinates of the boxes
    x0_1, y0_1, x0_2, y0_2 = box0
    x1_1, y1_1, x1_2, y1_2 = box1

    # Calculate the coordinates of the intersection rectangle
    x_inter1 = max(x0_1, x1_1)
    y_inter1 = max(y0_1, y1_1)
    x_inter2 = min(x0_2, x1_2)
    y_inter2 = min(y0_2, y1_2)

    # Calculate the area of the intersection rectangle
    inter_width = max(0, x_inter2 - x_inter1)
    inter_height = max(0, y_inter2 - y_inter1)
    inter_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    box0_area = (x0_2 - x0_1) * (y0_2 - y0_1)
    box1_area = (x1_2 - x1_1) * (y1_2 - y1_1)

    # Calculate the area of the union
    union_area = box0_area + box1_area - inter_area

    # Calculate the IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

class HumanDetector:
    def __init__(self):
        seg_cfg = get_cfg()
        seg_cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'))  #  "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        seg_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        seg_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml')  
        # model link: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn
        self.segmentation_predictor = DefaultPredictor(seg_cfg)
        self.segmentation_predictor = DefaultPredictor(seg_cfg)
        
        # Keypoint detection model
        pose_cfg = get_cfg()
        pose_cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))  # COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml
        pose_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        pose_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
        self.keypoints_predictor = DefaultPredictor(pose_cfg)
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
        print("DEBUG: human index:", human_index)
        seg_result_person = seg_result[human_index]
        # human related segmentation results
        seg_masks = seg_result_person.pred_masks.numpy()
        cv2.imwrite("debug_seg_masks.jpg", seg_masks[0]*255)  
        depth_masked = depth * seg_masks[0]
        plt.imshow(depth)
        plt.show()
        plt.imshow(depth_masked)
        plt.show()


        return depth_masked


# # open color map, depth map for obtaining human mask on depth:
# color_lst = os.listdir("example/realsense_color")
# depth_lst = os.listdir("example/realsense_depth")
# color_lst.sort()
# depth_lst.sort()
# color_paths = ["example/realsense_color/" + x for x in color_lst]
# depth_paths = ["example/realsense_depth/" + x for x in depth_lst]
# color = color_paths[100]
# depth = depth_paths[100]

# color = np.load(color)
# depth = np.load(depth)

# annotator = HumanDetector()
# results = annotator.forward(color, depth)

# depth_masked = np.zeros((depth.shape[0], depth.shape[1]))

# depth_masks = cv2.applyColorMap(cv2.convertScaleAbs(depth_masked, alpha=0.03), cv2.COLORMAP_JET)

# # Save the depth mask instead of displaying (for headless servers)
# output_path = "depth_mask_output.png"
# cv2.imwrite(output_path, depth_masks)
# print(f"Depth mask saved to: {output_path}")