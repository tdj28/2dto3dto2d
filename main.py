import glob
import os
import time
import multiprocessing
from helpers import setup_logger
from image_process import extract_frames, process_frame_to_ply, process_ply_to_image, process_frame_to_npz
from video_process import create_video
import sys
import torch

             
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def main():
    multiprocessing.set_start_method('spawn', force=True)

    input_video_path = './input_video.mp4'
    input_frames_path = './media/input_frames'
    output_frames_path = './media/output_frames'
    output_video_path = './output_video.mp4'
    frame_filename_glob_string = '%08d'

    logger = setup_logger('2dto3dto2d')



    media_folder = './media'
    input_frames_folder = os.path.join(media_folder, 'input_frames')
    output_frames_folder = os.path.join(media_folder, 'output_frames')
    ply_files_folder = os.path.join(media_folder, 'npz_files')

    ensure_directory_exists(media_folder)
    ensure_directory_exists(input_frames_folder)
    ensure_directory_exists(output_frames_folder)
    ensure_directory_exists(ply_files_folder)

    frame_extraction_complete = multiprocessing.Event()
    image_processing_complete = multiprocessing.Event()
    ply_extraction_complete = multiprocessing.Event()
    npz_extraction_complete = multiprocessing.Event()

    frame_queue = multiprocessing.Queue()
    ply_queue = multiprocessing.Queue()
    npz_queue = multiprocessing.Queue()

    # Start processes
    processes = [
        multiprocessing.Process(target=extract_frames, args=(input_video_path, input_frames_path, frame_extraction_complete)),
        multiprocessing.Process(target=process_frame_to_npz, args=(frame_queue, npz_queue, logger, npz_extraction_complete)),
        multiprocessing.Process(target=process_frame_to_npz, args=(frame_queue, npz_queue, logger, npz_extraction_complete)),
        multiprocessing.Process(target=process_frame_to_npz, args=(frame_queue, npz_queue, logger, npz_extraction_complete)),
        # multiprocessing.Process(target=process_ply_to_image, args=(ply_queue, frame_extraction_complete, ply_extraction_complete, image_processing_complete, logger)),
        # multiprocessing.Process(target=process_ply_to_image, args=(ply_queue, frame_extraction_complete, ply_extraction_complete, image_processing_complete, logger)),
        # multiprocessing.Process(target=process_ply_to_image, args=(ply_queue, frame_extraction_complete, ply_extraction_complete, image_processing_complete, logger)),
        # multiprocessing.Process(target=process_ply_to_image, args=(ply_queue, frame_extraction_complete, ply_extraction_complete, image_processing_complete, logger)),
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
    if not torch.cuda.is_available():
        sys.exit("Error: CUDA is not available.")
    main()
