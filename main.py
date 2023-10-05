import ffmpeg
import matplotlib
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from tqdm import tqdm  # For progress bar
import glob
import os
import time
import multiprocessing
import subprocess
import logging
import logging.handlers
import multiprocessing


def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,  # Set logging level to DEBUG to capture all messages
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logfile.log'),  # Log messages to a file
            logging.StreamHandler()  # Log messages to the console
        ]
    )

def extract_frames(video_path, frames_path, frame_extraction_complete):
    frame_filename_glob_string = '%08d'
    command = [
        'ffmpeg',
        '-i', video_path,
        f"{frames_path}/{frame_filename_glob_string}.png"
    ]
    process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
    returncode = process.wait()  # Wait for the process to complete
    if returncode != 0:
        stderr_output = process.stderr.read()
        error_message = f'ffmpeg error: {stderr_output}'
        print(error_message)
        raise Exception(error_message)  # Raise an exception with the error output
    frame_extraction_complete.set()


def process_frame_to_ply(frame_queue, ply_queue, logger):
    try:
        # Initialize the model
        print("initializing")
        logger.info("initializing")
        feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
        model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
        print("Model done")

    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error: {e}")
        raise
    # # Initialize the model
    # print("initializing")
    # feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    # print("and model")
    # model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    # print("Model done")
    while True:
        #print("True")
        frame_data = frame_queue.get()
        if frame_data is None:
            print("None")
            logger.info("Exit received in process_frame_to_ply.")
            break  # Exit when None is received
        frame_path, frame_num, num_frames = frame_data
        # ... process frame to ply
        #print(f"Processing frame {frame_num} of {num_frames}: {frame_path}")
        logger.info(f"Processing frame {frame_num}")
        #print("logged")
        image = Image.open(frame_path)
        #print("a")
        new_height = 480 if image.height > 480 else image.height
        new_height -= (new_height % 32)
        new_width = int(new_height * image.width / image.height)
        diff = new_width % 32
        #print("b")

        new_width = new_width - diff if diff < 16 else new_width + 32 - diff
        new_size = (new_width, new_height)
        image = image.resize(new_size)
        #print("bb")
        #print(type(image), image.size)
        inputs = feature_extractor(images=image, return_tensors="pt")

        #print("c")

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        #print("d")

        pad = 16
        output = predicted_depth.squeeze().cpu().numpy() * 1000.0
        output = output[pad:-pad, pad:-pad]
        image = image.crop((pad, pad, image.width - pad, image.height - pad))

        width, height = image.size
        #print(f"{width} {height}")
        depth_image = (output * 255 / np.max(output)).astype('uint8')
        image = np.array(image)
        depth_o3d = o3d.geometry.Image(depth_image)
        image_o3d = o3d.geometry.Image(image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

        ply_path = frame_path.replace('frames', 'ply').replace('.png', '.ply')
        #print(ply_path)
        success = o3d.io.write_point_cloud(ply_path, pcd)
        if success:
            logger.info(f"Successfully saved PLY file at {ply_path}")
        else:
            logger.error(f"Failed to save PLY file at {ply_path}")
        # 4. Camera Position Parameterization
        eye_x = np.interp(frame_num, [1, num_frames], [0, 0.13])
        eye_y = np.interp(frame_num, [1, num_frames], [-0.37, -0.44])
        eye_z = np.interp(frame_num, [1, num_frames], [-0.9, 0.6])
        ply_queue.put((ply_path, eye_x, eye_y, eye_z))  # Queue ply file and camera position for next stage


def process_ply_to_image(ply_queue, frame_extraction_complete, image_processing_complete, logger):

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
            fin_path = ply_path.replace('/ply/', '/final/').replace('.ply', '.png')
            print(fin_path)
            print("fin")
            try:
                pio.write_image(fig, fin_path)
            except Exception as e:
                logger.error(f"Error writing image: {e}")
                raise

            print("wrote")


        elif frame_extraction_complete.is_set():
            print("I'm broke")
            break  # Exit when frame extraction is complete and ply queue is empty
        else:
            time.sleep(1)  # Wait for more ply files to be queued
            print("sleep")
    image_processing_complete.set()  # Indicate that image processing is complete


def create_video(images_path, video_path):
    command = [
        'ffmpeg',
        '-framerate', '30',
        '-pattern_type', 'glob',
        '-i', f"{images_path}/*.png",
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        video_path
    ]
    process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
    returncode = process.wait()  # Wait for the process to complete
    if returncode != 0:
        stderr_output = process.stderr.read()
        error_message = f'ffmpeg error: {stderr_output}'
        print(error_message)
        raise Exception(error_message)  # Raise an exception with the error output

def main():
    input_video_path = './input_video.mp4'
    input_frames_path = './frames'
    output_frames_path = './final'
    output_video_path = './output_video.mp4'
    frame_filename_glob_string = '%08d'

    setup_logger()
    logger = logging.getLogger('depth2mesh')


    frame_extraction_complete = multiprocessing.Event()
    image_processing_complete = multiprocessing.Event()
    frame_queue = multiprocessing.Queue()
    ply_queue = multiprocessing.Queue()

    # Start processes
    processes = [
        multiprocessing.Process(target=extract_frames, args=(input_video_path, input_frames_path, frame_extraction_complete)),
        multiprocessing.Process(target=process_frame_to_ply, args=(frame_queue, ply_queue, logger)),
        # multiprocessing.Process(target=process_frame_to_ply, args=(frame_queue, ply_queue, logger)),  # Additional instance
        multiprocessing.Process(target=process_ply_to_image, args=(ply_queue, frame_extraction_complete, image_processing_complete, logger)),
        # multiprocessing.Process(target=process_ply_to_image, args=(ply_queue, frame_extraction_complete, image_processing_complete, logger)),
        ]
    for p in processes:
        p.start()

    # Monitor frame directory and queue new frames
    known_frames = set()
    frame_num = 0  # Initialize frame number

    while not frame_extraction_complete.is_set():
        current_frames = set(glob.glob(f"{input_frames_path}/*.png"))

        #print(current_frames)
        new_frames = current_frames - known_frames
        num_frames = len(current_frames)  # Update total number of frames
        for frame_path in new_frames:
            frame_num += 1  # Increment frame number
            frame_queue.put((frame_path, frame_num, num_frames))  # Updated line
            #print(frame_queue.qsize())
        #print(f"Queued {len(new_frames)} new frames")
        known_frames.update(new_frames)
        if frame_extraction_complete.is_set() and not new_frames:
            frame_queue.put(None)  # Signal ply processing to exit once all frames have been queued
        time.sleep(1)  # Wait before checking for new frames again



    # Signal ply processing to exit
    frame_queue.put(None)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    image_processing_complete.wait()
    # Create video from output images
    create_video(output_frames_path, output_video_path)

if __name__ == '__main__':
    main()
