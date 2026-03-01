from scipy.ndimage import label
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os
import argparse
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import traceback
import matplotlib.ticker as ticker
import pyvista as pv

# def mark_connected_components(depth_mask_persons):
#     # input: a 2-dimensional array
#     # each array is a depth map
#     # all elements in the depth map are 0s except the elements corresponding to human, where they are instead distances
#     # return:
#     # a human mask with background pixels all 0
#     # each connected components in the depth are assigned a unique label starting from 1
#     # the output is a 2-dimensional array
    
#     # example input;
#     # [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     # [0, 0, 3.0, 3.0, 2.9, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 3.1, 3.1, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 3.1, 3.2, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 1.2, 2.0, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 3.1, 3.2, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 2.9, 3.1, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
#     # example output:
#     # [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     # [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
#     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

def mark_connected_components(depth_mask_persons):
    """
    Label 8-connected non-zero regions in a 2D depth mask.
    Background (zeros) remain 0. Components are labeled 1..N.
    """
    arr = np.asarray(depth_mask_persons)
    if arr.ndim != 2:
        raise ValueError("depth_mask_persons must be a 2D array")
    # boolean mask of foreground (non-zero)
    mask = arr != 0
    # 8-connectivity structure
    structure = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]], dtype=bool)
    labeled, _ = label(mask, structure=structure)
    return labeled

def remove_small_regions(depth_map, min_size=100):
    """
    Remove small connected regions from the depth map.
    """
    labeled = mark_connected_components(depth_map)
    # Count the size of each region
    sizes = np.bincount(labeled.ravel())
    # Create a mask for regions to keep
    mask = sizes > min_size
    # Apply the mask to the labeled image
    cleaned = np.where(mask[labeled], labeled, 0)
    return cleaned

def merge_depth_masks(depth_maps):
    # give a list of depth maps with same shape
    # we assign their non-zero parts to previous maps' same positions.
    merged = depth_maps[0]
    for i in range(1, len(depth_maps)):
        merged = np.where(depth_maps[i] != 0, depth_maps[i], merged)
    return merged

def add_blender_infinite_grid(
    plotter: pv.Plotter,
    floor_z: float = 0.0,
    size: float = 200.0,
    fine_step: float = 0.5,
    coarse_step: float = 5.0,
    fade_radius: float = 40.0,
    center_xy: tuple = (0.0, 0.0),
    z_epsilon: float = 5e-3,
    fine_opacity: float = 0.35,
    coarse_opacity: float = 0.55,
    fine_line_width: float = 1.0,
    coarse_line_width: float = 2.0,
    max_resolution: int = 400,
    cmap_name: str = "gray",
    ):
    """
    Blender-like fading grid WITHOUT camera-follow (no callbacks; compatible with old PyVista).
    Returns dict with meshes/actors.
    """

    def _plane_resolution(step: float) -> int:
        target = max(2, int(round(size / max(step, 1e-6))))
        return int(np.clip(target, 2, max_resolution))

    def _make_plane(step, line_width, base_opacity):
        res = _plane_resolution(step)
        plane = pv.Plane(
            center=(center_xy[0], center_xy[1], floor_z + z_epsilon),
            direction=(0, 0, 1),
            i_size=size,
            j_size=size,
            i_resolution=res,
            j_resolution=res,
        )

        pts = plane.points
        r = np.sqrt((pts[:, 0] - center_xy[0])**2 + (pts[:, 1] - center_xy[1])**2)
        alpha = np.exp(-r / max(fade_radius, 1e-6)) * base_opacity
        alpha = np.clip(alpha, 0.0, 1.0)

        plane["alpha"] = alpha

        actor = plotter.add_mesh(
            plane,
            style="wireframe",
            scalars="alpha",
            opacity=alpha,              # per-point opacity
            cmap=cmap_name,
            show_scalar_bar=False,
            lighting=False,
            line_width=float(line_width),
        )
        return plane, actor

    fine_plane, fine_actor = _make_plane(fine_step, fine_line_width, fine_opacity)
    coarse_plane, coarse_actor = _make_plane(coarse_step, coarse_line_width, coarse_opacity)

    return {
        "fine_mesh": fine_plane,
        "coarse_mesh": coarse_plane,
        "fine_actor": fine_actor,
        "coarse_actor": coarse_actor,
    }
        
def add_camera_marker(
    p: pv.Plotter,
    pos=(0.0, 0.0, 1.0),          # (x,y,z)
    forward=(0.0, 1.0, 0.0),      # facing +y
    body_length=70.0,
    body_radius=70.0,
    color="#2b6cb0",
    arrow_scale=140.0,
    label="Camera",
    label_z_offset=70.0,        # vertical distance below camera
    ):
    pos = np.array(pos, dtype=float)
    fwd = np.array(forward, dtype=float)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)

    # ---- cone (camera body) ----
    cone_center = pos + 0.5 * body_length * fwd
    cone = pv.Cone(center=cone_center, direction=fwd, height=body_length, radius=body_radius)
    p.add_mesh(cone, color=color, smooth_shading=True, opacity=1.0)

    # ---- arrow (view direction) ----
    arrow = pv.Arrow(start=pos, direction=fwd, scale=arrow_scale)
    p.add_mesh(arrow, color=color, opacity=1.0)

    # # # ---- centered label below camera ----
    # label_pos = pos + np.array([-280.0, -30.0, -label_z_offset])

    # p.add_point_labels(
    #     [label_pos],
    #     [label],
    #     font_size=40,
    #     text_color='black',
    #     shape=False,          # no background box
    #     always_visible=True,
    # )

def plot_3d_point_cloud_new(
    point_cloud,
    max_num_persons,
    max_num_points,
    camera_height,
    labels,
    colors,
    threshold=0.5,
    point_size=5,
    show=True,
    return_plotter=False,
    floor_margin=0.0,
    grid_res=15,
    floor_z_mode="auto",  # "auto" | "zero" | float
    save=False,             
    save_name=None, 
    regularSpacing = True
    ):
    """
    New (PyVista) visualization that mimics a Unity-like scene:
    - beige floor plane
    - wireframe grid on floor
    - axes widget
    - translucent bounding 'room'
    - point clouds per person with per-person colors

    Keeps your old data format:
      point_cloud shape: (3, max_num_persons*(max_num_points+1))  (or compatible)
      last point of each person segment is an "indicator" scalar stored at point_cloud[0, indicator_idx]
    """

    # ---- sanity / format ----
    pc = np.asarray(point_cloud)
    # if pc.ndim != 2 or pc.shape[0] < 3:
    #     raise ValueError(f"point_cloud must be shaped like (3, M). Got {pc.shape}")

    points_per_person = max_num_points + 1

    all_points = []         # gather all valid points across persons for bounds
    person_clouds = []      # list of (points Nx3, color)

    if regularSpacing:
        for person_idx in range(max_num_persons):
            start_idx = person_idx * points_per_person
            end_idx = start_idx + points_per_person
            indicator_idx = end_idx - 1

            # indicator stored at point_cloud[0, indicator_idx]
            indicator_value = pc[0, indicator_idx]
            if indicator_value <= threshold:
                continue

            person_points = pc[:3, start_idx:end_idx]  # (3, points_per_person)

            x = person_points[0, :]
            y = person_points[1, :]
            z = person_points[2, :]

            # same validity rule as your old code
            valid = ~((x < 5) & (y < 5) & (z < 5) & (x > -5) & (y > -5) & (z > -5))
            x = x[valid]
            y = y[valid]
            z = z[valid]

            if x.size == 0:
                continue

            # same coordinate mapping as your old scatter:
            # old: ax.scatter(x_valid, z_valid, y_valid) with y flipped
            # => world coords: (X = x, Y = z, Z = -y)
            pts = np.stack([x, z, -y], axis=1)  # (N,3)

            # color: accept matplotlib-like or rgb/rgba
            c = colors[person_idx]
            person_clouds.append((pts, c))
            all_points.append(pts)
    else: # point_cloud is a list
        idx = 0
        for point_cloud_individual in point_cloud:
            person_clouds.append((point_cloud_individual, colors[idx]))
            all_points.append(point_cloud_individual)
            idx += 1

    # # If nothing to show
    # if len(all_points) == 0:
    #     if show:
    #         print("No valid persons/points above threshold to visualize.")
    #     return None

    # all_points = np.concatenate(all_points, axis=0)

    # ---- bounds / floor ----
    xmin, ymin, zmin = (-2000, 0, -1000*camera_height)
    xmax, ymax, zmax = (2000, 6000, 1000)

    dx, dy, dz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
    # avoid zero-size
    dx = dx if dx > 1e-6 else 1.0
    dy = dy if dy > 1e-6 else 1.0
    dz = dz if dz > 1e-6 else 1.0

    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    if floor_z_mode == "zero":
        floor_z = 0.0
    elif isinstance(floor_z_mode, (int, float)):
        floor_z = float(floor_z_mode)
    else:
        # auto: use min(0, zmin) so it often looks like a ground plane
        floor_z = min(0.0, float(zmin))




    # ---- build scene ----
    p = pv.Plotter(off_screen=True)
    p.set_background("#EBEBEB")
    
    # add_blender_infinite_grid(p, floor_z=0.0, follow_camera=True)
    add_blender_infinite_grid(p, 
                            floor_z=floor_z, 
                            center_xy=(cx, cy), 
                            size=100000,
                            fine_step = 150, 
                            coarse_step = 150, 
                            fade_radius=10000, 
                            fine_opacity = 0.08, 
                            coarse_opacity = 0.08,
                            cmap_name = 'gist_yarg'
                            )

    # Beige floor (like your screenshot)
    floor = pv.Plane(
        center=(cx, cy, floor_z),
        direction=(0, 0, 1),
        i_size=dx * (1 + floor_margin),
        j_size=dy * (1 + floor_margin),
        i_resolution=1,
        j_resolution=1,
    )
    p.add_mesh(floor, color="#DCC9B5", opacity=0.7, lighting=True)

    sphere = pv.Sphere(
        radius=20,
        theta_resolution=8,
        phi_resolution=8,
    )

    for i, (pts, c) in enumerate(person_clouds):
        cloud = pv.PolyData(pts)

        glyphs = cloud.glyph(
            geom=sphere,
            scale=False,
            orient=False,
        )

        p.add_mesh(
            glyphs,
            color=c,
            opacity=1.0,
            smooth_shading=True,
        )
        

    bounds = (xmin, xmax, ymin, ymax, min(floor_z, zmin), 1000)
    box = pv.Box(bounds=bounds)
    # Extract faces and filter out the bottom face (x-y plane at minimum z)
    # Get all faces from the box
    faces = box.extract_surface()
    face_centers = faces.cell_centers()
    bottom_z = min(floor_z, zmin)
    # Filter out the face at the bottom (face center z-coordinate is at bottom_z)
    visible_faces = []
    for i in range(faces.n_cells):
        center = face_centers.points[i]
        # Keep faces that are not at the bottom z-level (with small tolerance)
        if abs(center[2] - bottom_z) > 1e-6:
            visible_faces.append(i)
    visible_faces = [0, 1, 3]
    if len(visible_faces) > 0:
        visible_box = faces.extract_cells(visible_faces)
        p.add_mesh(visible_box, color="white", opacity=0.6)        # translucent faces
    # p.add_mesh(box.outline(), color="gray", line_width=2)   # outline

    # Nice axes widget (bottom-left)
    # p.add_axes(line_width=3)
    

    # Camera similar to isometric tilt
    p.camera_position = "iso"
    p.camera_position = [
        (0.5 * (xmin + xmax), -400, 600),   # camera location: center of x-z plane
        # (0.5 * (xmin + xmax), -30, 30),   # camera location: center of x-z plane
        (0, 2000, 0),    # focal point (center of the scene)
        (0, 0, 1)     # view-up direction (z axis is up)
    ]
    z_lower = -1000 * float(camera_height)
    z_upper = 1000.0
    # Expand bounds to respect that z range
    bounds2 = (xmin, xmax, ymin, ymax, z_lower, z_upper)
    p.reset_camera(bounds=bounds2)
    p.camera.zoom(1.4)

    # -------------------- SAVE TO PDF --------------------
    if save and save_name is not None:
        if not save_name.lower().endswith(".pdf"):
            save_name = save_name + ".pdf"
        # Render once before saving (required by VTK)
        p.show(auto_close=False, interactive=False)
        try:
            p.save_graphic(save_name)
            print(f"[PyVista] Saved scene to PDF: {save_name}")
        except Exception as e:
            print(f"[PyVista] PDF export failed: {e}")
    if show:
        p.show()
    else:
        if save:
            p.close()

    if return_plotter:
        return p
    return p.screenshot(return_img=True)


if __name__=="__main__":
    # data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 3.0, 3.0, 2.9, 0, 0, 0, 0, 0],
    # [0, 0, 0, 3.1, 3.1, 0, 0, 0, 0, 0],
    # [0, 0, 0, 3.1, 3.2, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [13, 0, 11.1, 13, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 12, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 1.2, 2.0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 3.1, 3.2, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 2.9, 3.1, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # data1 =  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [13, 0, 11.1, 13, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 12, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 2.9, 3.1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # data = np.array(data)

    # print(mark_connected_components(data))
    import pickle as pkl 

    with open("/home/zx/Desktop/zx/TAP3D_demo/data/20260227180229779364/annotation/20260227180246713069.pkl", "rb") as f:
        data = pkl.load(f)

    labels1 = [f'Pred. P{i+1}' for i in range(6)]
    colors1 = plt.colormaps.get_cmap('Set1')(np.linspace(0, 1, 6))

    gt_ptcloud = data["point_cloud_person"]
    img = plot_3d_point_cloud_new(gt_ptcloud, -1, -1, 1.3, labels1, colors=colors1, regularSpacing=False, show=False)
    plt.imshow(img)
    plt.show()
    
    #image2 = plot_3d_point_cloud_new(ptcloud.cpu().numpy(),  exp_config['max_num_persons'], exp_config['max_num_points'], camera_height=1.3, labels=labels1, colors=colors1)
    #image1 = plot_3d_point_cloud_new(pcl_gt.T, 1, pcl_gt.shape[0]-1, camera_height=1.3, labels=labels1, colors=colors1)
                
                



        
        
    # def visualize_gt_pcd(self, ax, result_dict, use_old_plot = False):
    #     if use_old_plot:
    #         ax.clear()
    #         pcl_gt =  concat_pcd(result_dict)
    #         if pcl_gt is not None:
    #             print("DEBUG: shape is:", pcl_gt.shape)
    #             plot_3d_point_cloud(fig, ax, pcl_gt.T, 1, pcl_gt.shape[0]-1)
    #         else:
    #             plot_3d_point_cloud(fig, ax, np.zeros([3, 1*42]), 1, 42-1)
    #         fig.canvas.draw()
    #         # fig.canvas.flush_events()
    #         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    #         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         image = cv2.resize(image, (960, int(960 * image.shape[0] / image.shape[1])))
    #     else: 
    #         pcl_gt =  concat_pcd(result_dict)
    #         labels1 = [f'Pred. P{i+1}' for i in range(6)]
    #         colors1 = plt.colormaps.get_cmap('Set1')(np.linspace(0, 1, 6))
    #         if pcl_gt is None:
    #             pcl_gt = np.zeros([1*6006, 3])
    #         image = plot_3d_point_cloud_new(pcl_gt.T,  exp_config['max_num_persons'], exp_config['max_num_points'], camera_height=1.3, labels=labels1, colors=colors1)
    #         image = cv2.resize(image, (960, int(960 * image.shape[0] / image.shape[1])))
    #         print('image shape: ', image.shape)
    #     return image

    # # pcl_gt =  concat_pcd(result_dict)
    # #             if pcl_gt is not None:
    # #                 print("DEBUG: shape is:", pcl_gt.shape)
    # #                 # plot_3d_point_cloud(fig, ax, pcl_gt.T, 1, pcl_gt.shape[0]-1)
    # #                 image1 = plot_3d_point_cloud_new(pcl_gt.T, 1, pcl_gt.shape[0]-1, camera_height=1.3, labels=labels1, colors=colors1)
                

        
    # # ================== for visualization of point clouds: 2 axes for inference and annotate, 1 axis for annotate, no axis for collection ============
    # def visualize_pred_pcd(self, ax1, ptcloud, exp_config, use_old_plot = False):
    #     if use_old_plot:
    #         ax1.clear()
    #         # print(ptcloud.cpu().numpy().shape, "DDDDEBUG")
    #         plot_3d_point_cloud(fig, ax1, ptcloud,  exp_config['max_num_persons'], exp_config['max_num_points'])
            
    #         fig.canvas.draw()
    #         # fig.canvas.flush_events()
    #         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    #         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #         # rescale image such that its width is 960, and its height-width ration remains unchanged
    #         cv2.putText(image, f"Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #         cv2.putText(image, f"Prediction", (10 + 960, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #         image = cv2.resize(image, (960*2, int(960 * 2 * image.shape[0] / image.shape[1])))
    #     else:
    #         labels1 = [f'Pred. P{i+1}' for i in range(6)]
    #         colors1 = plt.colormaps.get_cmap('Set1')(np.linspace(0, 1, 6))
    #         image = plot_3d_point_cloud_new(ptcloud,  exp_config['max_num_persons'], exp_config['max_num_points'], camera_height=1.3, labels=labels1, colors=colors1)
    #         image = cv2.resize(image, (960, int(960 * image.shape[0] / image.shape[1])))
    #     return image