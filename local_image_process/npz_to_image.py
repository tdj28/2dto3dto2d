import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import time
import traceback
from helpers import setup_logger
import io
from PIL import Image

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import numpy as np

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    FoVPerspectiveCameras
)
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, PointsRenderer, AlphaCompositor
import matplotlib.pyplot as plt


MAX_RETRIES = 3

def process_ply_to_image(ply_queue, frame_extraction_complete, ply_extraction_complete, image_processing_complete, logger2):
    logger = setup_logger('2dto3dto2d')
    while True:
        if not ply_queue.empty():
            print(ply_queue.qsize())
            ply_data = ply_queue.get()
            ply_path, eye_x, eye_y, eye_z = ply_data

            # Load the PLY file for plotting
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) * 255  # Scale color values to [0, 255]
            colors_str = ["rgb({}, {}, {})".format(r, g, b) for r, g, b in colors]
            print("fig")
            fig = go.Figure(data=[
              go.Scatter3d(
                      x=points[:, 0],
                      y=points[:, 1],
                      z=points[:, 2],
                      mode='markers',
                      marker=dict(
                          size=2,
                          color=colors_str,
                          opacity=1.0
                      )
                  )
              ])

            fig.update_layout(
                width=700,
                height=500,
                scene=dict(
                    aspectmode='cube',
                    bgcolor='black',
                    xaxis=dict(visible=True),
                    yaxis=dict(visible=True),
                    zaxis=dict(visible=True),
                ),
                scene_camera=dict(
                    eye=dict(x=eye_x, y=eye_y, z=eye_z),
                    up=dict(x=0, y=-1, z=0),
                    center=dict(x=0, y=-0.1, z=1)
                ),
                margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=0
                )
            )

            # Save the output image
            fin_path = ply_path.replace('ply_files', 'output_frames').replace('.ply', '.png')
            print(fin_path)
            print("fin")
            for i in range(MAX_RETRIES):
                try:
                    pio.write_image(fig, fin_path)
                except FileNotFoundError as e:
                    logger.error(f"File not found: {fin_path}. Error: {e}")
                    logger.debug(traceback.format_exc())  # Log the full traceback
                    break  # No point in retrying if file not found
                except IOError as e:
                    logger.error(f"IO error writing image to: {fin_path}. Error: {e}. Attempt: {i+1}")
                    logger.debug(traceback.format_exc())  # Log the full traceback
                    if i < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                        time.sleep(1)  # Wait for a while before retrying
                    continue  # Retry
                except Exception as e:
                    logger.error(f"Unexpected error writing image to: {fin_path}. Error: {e}")
                    logger.debug(traceback.format_exc())  # Log the full traceback
                    break  # No point in retrying if we don't know what the error is
                else:
                    print("wrote")
                    break  # Exit the loop if write was successful



        elif frame_extraction_complete.is_set() and ply_extraction_complete.is_set():
            print("I'm broke in ply_to_image.py")
            break  # Exit when frame extraction is complete and ply queue is empty
        else:
            time.sleep(1)  # Wait for more ply files to be queued
            print("sleep")
    image_processing_complete.set()  # Indicate that image processing is complete

def process_npz_to_image(npz_queue, output_image_queue, frame_extraction_complete, npz_extraction_complete, image_processing_complete):
    
    logger = setup_logger('2dto3dto2d')

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    while True:
        if not npz_queue.empty():
            npz_data = npz_queue.get()
            points, colors, frame_index = npz_data
            pointcloud = Pointclouds(points=[points], features=[colors])
            # Split by "/" and take the last part. Then strip off ".npz" from the end.
            #number_string = npz_path.split("/")[-1].replace(".npz", "")
            # Convert to integer to get rid of leading zeros.
            #number = int(number_string)
            # Load point cloud
            #pointcloud = np.load(npz_path)
            verts = torch.Tensor(pointcloud['points']).to(device)
            rgb = torch.Tensor(pointcloud['colors']).to(device)

            # Compute centroid and bounding box
            centroid = verts.mean(dim=0)
            min_vals, _ = verts.min(dim=0)
            max_vals, _ = verts.max(dim=0)

            # Normalize the point cloud
            scale = (max_vals - min_vals).max().item()
            normalized_verts = (verts - centroid) / scale

            # Initialize a camera close to the centroid
            R, T = look_at_view_transform(dist=-1, elev=0, azim=number)
            R[0, 1] = -R[0, 1]  # Flip the y-axis

            cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

            # Define rasterization settings
            raster_settings = PointsRasterizationSettings(
                image_size=512, 
                radius = 0.005,  # Adjusted for normalized point cloud
                points_per_pixel = 10
            )

            # Create a points renderer
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            renderer = PointsRenderer(
                rasterizer=rasterizer,
                compositor=AlphaCompositor()
            )

            # Render normalized point cloud
            point_cloud = Pointclouds(points=[normalized_verts], features=[rgb])
            images = renderer(point_cloud)
            plt.figure(figsize=(10, 10))
            plt.imshow(images[0, ..., :3].cpu().numpy())
            plt.axis("off")
            #fin_path = npz_path.replace('npz_files', 'output_frames').replace('.npz', '.png')
            #frame_index = 0
            buf = io.BytesIO()
            plt.savefig(buf, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            buf.seek(0)
            output_image_queue.put((frame_index, buf))
        elif frame_extraction_complete.is_set() and npz_extraction_complete.is_set():
            print("I'm done with everything")
            break  # Exit when frame extraction is complete and ply queue is empty
        else:
            time.sleep(1)  # Wait for more ply files to be queued
    image_processing_complete.set()  # Indicate that image processing is complete

