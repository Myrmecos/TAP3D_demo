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


def plot_3d_point_cloud_new(
    point_cloud,
    max_num_persons,
    max_num_points,
    camera_height,
    labels,
    colors,
    threshold=0.5,
    point_size=10,
    show=True,
    return_plotter=False,
    floor_margin=0.25,
    grid_res=25,
    floor_z_mode="auto",  # "auto" | "zero" | float
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
    if pc.ndim != 2 or pc.shape[0] < 3:
        raise ValueError(f"point_cloud must be shaped like (3, M). Got {pc.shape}")

    points_per_person = max_num_points + 1

    all_points = []         # gather all valid points across persons for bounds
    person_clouds = []      # list of (points Nx3, color)

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

    # If nothing to show
    if len(all_points) == 0:
        if show:
            print("No valid persons/points above threshold to visualize.")
        return None

    all_points = np.concatenate(all_points, axis=0)

    # ---- bounds / floor ----
    xmin, ymin, zmin = all_points.min(axis=0)
    xmax, ymax, zmax = all_points.max(axis=0)

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
    p = pv.Plotter()
    p.set_background("white")

    # Beige floor (like your screenshot)
    floor = pv.Plane(
        center=(cx, cy, floor_z),
        direction=(0, 0, 1),
        i_size=dx * (1 + floor_margin),
        j_size=dy * (1 + floor_margin),
        i_resolution=1,
        j_resolution=1,
    )
    p.add_mesh(floor, color="#e8d7c3", opacity=1.0, lighting=False)

    # Wireframe grid on the floor (a subdivided plane as wireframe)
    grid_plane = pv.Plane(
        center=(cx, cy, floor_z + 1e-4),
        direction=(0, 0, 1),
        i_size=dx * (1 + floor_margin),
        j_size=dy * (1 + floor_margin),
        i_resolution=grid_res,
        j_resolution=grid_res,
    )
    p.add_mesh(grid_plane, style="wireframe", color="gray", opacity=0.35, line_width=1)

    # Add each person’s points
    for i, (pts, c) in enumerate(person_clouds):
        cloud = pv.PolyData(pts)

        # allow matplotlib RGBA arrays etc.
        # If you pass a named color string that's also fine.
        p.add_points(
            cloud,
            color=c,
            render_points_as_spheres=True,
            point_size=point_size,
            opacity=0.6,
        )

    # Translucent "room"/bounding box (like a faint cube)
    bounds = (xmin, xmax, ymin, ymax, min(floor_z, zmin), zmax)
    box = pv.Box(bounds=bounds)
    p.add_mesh(box, color="lightgray", opacity=0.08)        # translucent faces
    p.add_mesh(box.outline(), color="gray", line_width=2)   # outline

    # Nice axes widget (bottom-left)
    p.add_axes(line_width=3)

    # Camera similar to isometric tilt
    p.camera_position = "iso"

    # Optional: keep Z limits similar to your old z-lim idea
    # (This is mainly for consistent framing if your camera_height matters)
    # You can also remove this if you prefer auto framing.
    z_lower = -1000 * float(camera_height)
    z_upper = 1000.0
    # Expand bounds to respect that z range
    bounds2 = (xmin, xmax, ymin, ymax, z_lower, z_upper)
    p.reset_camera(bounds=bounds2)

    if show:
        p.show()

    if return_plotter:
        return p
    return None