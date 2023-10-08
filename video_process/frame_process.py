import subprocess
from helpers import setup_logger
import cv2
import numpy as np
import os
import psutil
import time

def extract_frames_ffmpeg(
        video_path,
        frames_path,
        frame_extraction_complete):
    logger = setup_logger('2dto3dto2d')
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


def extract_frames(
        video_path,
        input_img_queue,
        frame_extraction_complete,
        write_to_file=False,
        output_dir=None):
    logger = setup_logger('2dto3dto2d:extract_frames')
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_index = 0
        while True:
            while True:
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 80:
                    logger.info(f"Memory usage: {memory_info.percent}%. Sleeping for 5 seconds.")
                    time.sleep(5)  # sleep for 5 seconds, hope to let other processes catch up
                else:
                    break  # if enough memory is available, break
            ret, frame = cap.read()
            if not ret:
                break
            outfile_path = None
            if write_to_file:
                if output_dir is None:
                    raise ValueError("output_dir must be specified when write_to_file is True")
                filename = os.path.join(output_dir, f"frame_{frame_index:08d}.png")
                outfile_path = filename
                cv2.imwrite(filename, frame)
                frame = None
                logger.info(f"Wrote frame {frame_index} to file {filename}")

            input_img_queue.put((frame_index, total_frames, fps, frame, write_to_file, outfile_path))
            logger.info(f"Queued frame {frame_index}")
            frame_index += 1
        cap.release()
    except Exception as e:
        logger.error(f"Error in extract_frames: {e}")
    frame_extraction_complete.set()