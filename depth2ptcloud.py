import trimesh
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class DepthMask2PointCloudFast(nn.Module):
    '''
    Optimized version of DepthMask2PointCloud with faster processing
    '''
    def __init__(self, exp_config):
        super(DepthMask2PointCloudFast, self).__init__()
        self.sensor_name = exp_config['sensor_name']
        self.max_num_points = exp_config['max_num_points']
        self.max_num_persons = exp_config['max_num_persons']
        
        # Pre-compute camera parameters
        if self.sensor_name == 'senxor_m08':
            self.hfov_deg = 90
            self.vfov_deg = 67
            self.height = 62
            self.width = 80
        elif self.sensor_name == 'senxor_m16':
            self.hfov_deg = 88.5
            self.vfov_deg = 58.5
            self.height = 120
            self.width = 160
        elif self.sensor_name == 'seek_thermal':
            self.hfov_deg = 81
            self.vfov_deg = 59
            self.height = 150
            self.width = 200
        else:
            raise ValueError(f"Sensor name {self.sensor_name} not supported")
        
        # Pre-compute camera intrinsics
        self._setup_camera_intrinsics()
        
    def _setup_camera_intrinsics(self):
        """Pre-compute camera intrinsics for faster point cloud conversion"""
        hfov_rad = np.deg2rad(self.hfov_deg)
        vfov_rad = np.deg2rad(self.vfov_deg)
        
        # Calculate focal lengths
        self.fx = self.width / (2 * np.tan(hfov_rad / 2))
        self.fy = self.height / (2 * np.tan(vfov_rad / 2))
        
        # Pre-compute normalized pixel coordinates
        x = np.arange(self.width)
        y = np.arange(self.height)
        x, y = np.meshgrid(x, y)
        
        # Normalize to camera coordinates
        self.x_cam = (x - self.width / 2) / self.fx
        self.y_cam = (y - self.height / 2) / self.fy
        
    def forward(self, depth_mask_3C):
        '''
        Args:
            depth_mask_3C: torch.Tensor of shape (B, 3, H, W)
        Returns:
            point_cloud_3C: Tensor of shape [B, 3, (max_num_points+1)*max_num_persons]
        '''
        batch_size = depth_mask_3C.shape[0]
        device = depth_mask_3C.device
        dtype = depth_mask_3C.dtype
        
        # Initialize output tensor
        point_cloud_3C = torch.zeros(batch_size, 3, (self.max_num_points+1)*self.max_num_persons, 
                                   dtype=dtype, device=device)
        
        # Process all batches at once to minimize CPU-GPU transfers
        depth_maps = depth_mask_3C[:, 0]  # (B, H, W)
        indicator_maps = torch.round(depth_mask_3C[:, 1])  # (B, H, W)
        
        # Convert to numpy only once
        depth_maps_np = depth_maps.detach().cpu().numpy()
        indicator_maps_np = indicator_maps.detach().cpu().numpy()
        
        for b in range(batch_size):
            depth_map = depth_maps_np[b]
            indicator_map = indicator_maps_np[b]
            
            # Process all persons in this batch simultaneously
            self._process_batch_persons(depth_map, indicator_map, b, point_cloud_3C, device, dtype)
            
        return point_cloud_3C
    
    def _process_batch_persons(self, depth_map, indicator_map, batch_idx, point_cloud_3C, device, dtype):
        """Process all persons in a batch efficiently"""
        # Find unique person indices (excluding 0)
        indicator_map = indicator_map.astype(int)
        unique_persons = np.unique(indicator_map)
        unique_persons = unique_persons[unique_persons > 0]
        
        for person_idx in unique_persons:
            # Convert numpy scalar to Python int for tensor slicing
            person_idx = int(person_idx)
            if person_idx > self.max_num_persons:
                continue
                
            # Create person mask
            person_mask = (indicator_map == person_idx)
            
            if not np.any(person_mask):
                continue
            
            # Extract person depth with mask
            person_depth = depth_map * person_mask
            
            # Fast point cloud conversion and processing
            points = self._fast_point_cloud_processing(person_depth, person_mask)
            
            if points is not None:
                # Calculate offset and store results
                offset = (person_idx - 1) * (self.max_num_points + 1)
                point_cloud_3C[batch_idx, :, offset:offset + self.max_num_points] = torch.tensor(
                    points, dtype=dtype, device=device)
                # Add indicator point
                point_cloud_3C[batch_idx, 0, offset + self.max_num_points] = 1.0
    
    def _fast_point_cloud_processing(self, person_depth, person_mask):
        """Fast point cloud processing with vectorized operations"""
        # Apply depth threshold and mask
        valid_mask = (person_depth > 3) & person_mask
        
        if not np.any(valid_mask):
            return None
        
        # Vectorized point cloud generation
        z_cam = person_depth
        x_3d = self.x_cam * z_cam
        y_3d = self.y_cam * z_cam
        
        # Stack coordinates
        point_cloud = np.stack((x_3d, y_3d, z_cam), axis=-1)
        
        # Extract valid points using boolean indexing
        valid_points = point_cloud[valid_mask]
        
        if valid_points.shape[0] == 0:
            return None
        
        # Fast outlier removal using vectorized operations
        valid_depths = person_depth[valid_mask]
        if valid_depths.shape[0] == 0:
            return None
            
        # Calculate outlier bounds
        lower_bound, upper_bound = self._fast_outlier_removal(valid_depths)
        
        # Apply outlier filter
        outlier_mask = (valid_depths >= lower_bound) & (valid_depths <= upper_bound)
        filtered_points = valid_points[outlier_mask]
        
        if filtered_points.shape[0] == 0:
            return None
        
        # Fast sampling
        return self._fast_point_cloud_sampling(filtered_points)
    
    def _fast_outlier_removal(self, depths):
        """Fast outlier removal using numpy operations"""
        if len(depths) == 0:
            return 0, float('inf')
        
        # Use numpy's percentile for faster computation
        Q1 = np.percentile(depths, 25)
        Q3 = np.percentile(depths, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return lower_bound, upper_bound
    
    def _fast_point_cloud_sampling(self, point_cloud):
        """Fast point cloud sampling with optimized operations"""
        num_points = point_cloud.shape[0]
        
        if num_points < self.max_num_points:
            # Pad with zeros
            pad_size = self.max_num_points - num_points
            padded_points = np.zeros((self.max_num_points, 3))
            padded_points[:num_points] = point_cloud
            return padded_points.T
        else:
            # Random sampling without replacement
            indices = np.random.choice(num_points, self.max_num_points, replace=False)
            return point_cloud[indices].T


def plot_3d_point_cloud(point_cloud, max_num_persons, max_num_points, camera_height=1, elev=15, azim=-45, ax=None, threshold=0.5, s= 10):
    points_per_person = max_num_points + 1
    scatter_ret = None
    
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
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
        #print(f"point_cloud shape: {point_cloud.shape}")
        #print(f"point_cloud indicator point at point_cloud[0, {indicator_idx}]: {point_cloud[0, indicator_idx]}")
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
 
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    # set the axis limit
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(0, 4000)
    ax.set_zlim(-1000*camera_height, 1000)
    # Add legend
    # if max_num_persons > 0:
    #     ax.legend()
    # set the view angle
    ax.view_init(elev, azim)
    if ax is None:
        plt.tight_layout()
        plt.show()    
    return None