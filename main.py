import glob
import os
import time
import multiprocessing
from helpers import setup_logger
from local_image_process import process_frame_to_npz, process_npz_to_image
from video_process import create_video, extract_frames, collect_and_write_images
import sys
import torch

             
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def main():
    
    try:
        multiprocessing.set_start_method('spawn', force=True)

        logger = setup_logger('2dto3dto2d')
        logger.info("Starting main function")


        write_to_file = False
        output_dir = None

        ensure_directory_exists('./media')
        logger.debug("Ensured media directory exists")


        input_video_path = './media/input_video.mp4'
        output_video_path = './media/output_video.mp4'

        # Validate inputs
        if not os.path.isfile(input_video_path):
            logger.error(f"Input video file does not exist: {input_video_path}")
            return
        else:
            logger.info(f"Input video file exists: {input_video_path}")


        num_cpus = multiprocessing.cpu_count()
        logger.info(f"Number of CPUs: {num_cpus}")


        frame_extraction_complete = multiprocessing.Event()
        npz_extraction_complete = [multiprocessing.Event() for _ in range(max(1, num_cpus//2))]
        image_processing_complete = [multiprocessing.Event() for _ in range(max(1, num_cpus//2))]
        final_video_creation_complete = multiprocessing.Event()

        input_img_queue = multiprocessing.Queue()
        npz_queue = multiprocessing.Queue()
        output_image_queue = multiprocessing.Queue()

        # Start processes
        processes = [
            multiprocessing.Process(target=extract_frames, args=(input_video_path, input_img_queue, frame_extraction_complete, write_to_file, output_dir)),
        ]

        #for i in range(num_cpus):
        for i in range(max(1, num_cpus//2)):
            processes.append(
                multiprocessing.Process(
                    target=process_frame_to_npz,
                    args=(
                        input_img_queue,
                        npz_queue,
                        frame_extraction_complete,
                        npz_extraction_complete[i]
                        )))
            

            processes.append(
                multiprocessing.Process(
                    target=process_npz_to_image,
                    args=(
                        npz_queue,
                        output_image_queue,
                        npz_extraction_complete,
                        image_processing_complete[i]
                        )))

        processes.append(multiprocessing.Process(target=collect_and_write_images, args=(output_image_queue, image_processing_complete, final_video_creation_complete, output_video_path)))

        for p in processes:
            p.start()

        final_video_creation_complete.wait()

        # Wait for all processes to finish
        for p in processes:
            p.join()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return


if __name__ == '__main__':
    if not torch.cuda.is_available():
        sys.exit("Error: CUDA is not available.")
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
