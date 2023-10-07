import subprocess
import cv2
import numpy as np
import time
from PIL import Image
from helpers import setup_logger

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

def collect_and_write_images(output_image_queue, frame_extraction_complete, npz_extraction_complete, image_processing_complete, final_video_creation_complete, output_video_path):
    logger = setup_logger('2dto3dto2d:collect_and_write_images')
    logger.info("Starting to collect and write images to final video.")
    images = []
    try:
        while True:
            if not output_image_queue.empty():
                frame_index, image_buf, write_to_file, fin_path = output_image_queue.get()
                
                try:
                    if write_to_file:
                        image = np.array(Image.open(fin_path))
                    else:
                        image = np.array(Image.open(image_buf))
                except Exception as e:
                    logger.error(f"Error opening image at frame {frame_index}: {e}")
                    continue

                images.append((frame_index, image))
                logger.info(f"Collected image {frame_index}")
            elif frame_extraction_complete.is_set() and npz_extraction_complete.is_set() and image_processing_complete.is_set():
                break
            else:
                time.sleep(1)  # Wait for more images to be queued

        # Sort images by frame_index
        images.sort(key=lambda x: x[0])

        # Get image dimensions from the first image
        height, width, _ = images[0][1].shape

        # Initialize video writer
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
            video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
        except Exception as e:
            logger.error(f"Error initializing video writer: {e}")
            raise
        # Write images to video
        for i, image in enumerate(images):
            try:
                video.write(image)
                logger.info(f"Successfully wrote frame {i} to video.")
            except Exception as e:
                logger.error(f"Error writing frame {i} to video: {e}")
                continue

        video.release()
        logger.info(f"Video written to {output_video_path}")
    except Exception as e:
        logger.error(f"Error during video creation: {e}")
        raise
    finally:
        final_video_creation_complete.set()
        logger.info("Finished collecting and writing images to video.")