import subprocess
from helpers import setup_logger
import cv2
import numpy as np

def extract_frames_ffmpeg(video_path, frames_path, frame_extraction_complete):
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

def extract_frames(video_path, input_img_queue, frame_extraction_complete):
    logger = setup_logger('2dto3dto2d')
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            input_img_queue.put((frame_index, total_frames, frame))
            frame_index += 1
        cap.release()
    except Exception as e:
        logger.error(f"Error in extract_frames: {e}")
