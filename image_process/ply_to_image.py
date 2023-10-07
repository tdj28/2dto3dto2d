import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import time
import traceback
from helpers import setup_logger

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

