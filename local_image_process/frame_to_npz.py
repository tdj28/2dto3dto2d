from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import open3d as o3d
import torch
import numpy as np
from PIL import Image
from helpers import setup_logger
import os

def initialize_model(logger):
    try:
        print("initializing")
        # Determine model loading path
        if os.environ.get("INSIDE_DOCKER") == "1":
            model_path = "/app/glpn-nyu"  # or wherever you store it in the Docker image
        else:
            model_path = "vinvino02/glpn-nyu"
        feature_extractor = GLPNImageProcessor.from_pretrained(model_path)
        model = GLPNForDepthEstimation.from_pretrained(model_path)
        print("Model done")
        return feature_extractor, model
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error: {e}")
        return None, None
    
def process_frame_to_ply(frame_queue, ply_queue, logger, ply_extraction_complete):
    logger = setup_logger('2dto3dto2d')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor, model = initialize_model(logger)
    model = model.to(device)


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
        inputs = inputs.to(device)
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

        ply_path = frame_path.replace('./media/input_frames', './media/ply_files').replace('.png', '.ply')
        #print(ply_path)
        success = o3d.io.write_point_cloud(ply_path, pcd)
        if success:
            logger.info(f"Successfully saved PLY file at {ply_path}")
        else:
            logger.error(f"Failed to save PLY file at {ply_path}")
        # 4. Camera Position Parameterization
        eye_x = 0 # np.interp(frame_num, [1, num_frames], [0, 0])
        eye_y = -0.4 #np.interp(frame_num, [1, num_frames], [-0.37, -0.44])
        eye_z = -0.8 #np.interp(frame_num, [1, num_frames], [-0.9, 0.6])
        ply_queue.put((ply_path, eye_x, eye_y, eye_z))  # Queue ply file and camera position for next stage
    ply_extraction_complete.set()


def process_frame_to_npz(
        input_img_queue,
        npz_queue,
        frame_extraction_complete,
        npz_extraction_complete):
     

    try:

        logger = setup_logger('2dto3dto2d:process_frame_to_npz')
        logger.info("Initializing model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor, model = initialize_model(logger)
        model = model.to(device)

        while True:
            try:
                if frame_extraction_complete.is_set() and input_img_queue.empty():
                    break
                
                frame_index, total_frames, fps, frame_data, write_to_file, outfile_path = input_img_queue.get()
                logger.debug(f"Got frame {frame_index} from queue for converstion to NPZ obj")
                if frame_data is None and write_to_file is False:
                    logger.info("Exit received in process_frame_to_npz, frame_data is None but write_to_file is False.")  # Changed log message
                    break  
                # frame_path, frame_num, num_frames = frame_data
                #logger.info(f"Processing frame {frame_num}")
                if write_to_file:
                    image = Image.open(outfile_path)
                else:
                    image = Image.fromarray(frame_data)
                logger.debug(f"Loaded image: {frame_index} from queue for converstion to NPZ obj")

                new_height = 480 if image.height > 480 else image.height
                new_height -= (new_height % 32)
                new_width = int(new_height * image.width / image.height)
                diff = new_width % 32

                new_width = new_width - diff if diff < 16 else new_width + 32 - diff
                new_size = (new_width, new_height)
                image = image.resize(new_size)

                inputs = feature_extractor(images=image, return_tensors="pt")
                inputs = inputs.to(device)

                with torch.no_grad():
                    try:
                        outputs = model(**inputs)
                        predicted_depth = outputs.predicted_depth
                    except Exception as e:
                        logger.error(f"Error in model inference: {e}")
                        continue

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

                # Extract point coordinates and colors from Open3D point cloud
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                
                # Save to npz
                npz_path = None
                if write_to_file:
                    try:
                        npz_path = outfile_path.replace('png', 'npz')
                        logger.info(f"Writing NPZ file to {npz_path}")
                        np.savez(npz_path, points=points, colors=colors)
                        logger.info(f"Successfully saved NPZ file at {npz_path}")
                        points = None
                        colors = None
                    except Exception as e:
                        logger.error(f"Error saving NPZ file at {npz_path}: {e}")
                        continue

                try:
                    npz_queue.put((points, colors, frame_index, total_frames, fps, write_to_file, npz_path))
                    logger.info(f"Queued (npz_queue) frame {frame_index}")
                except Exception as e:
                    logger.error(f"Error queuing (npz_queue) frame {frame_index}: {e}")
                    continue
        
            except Exception as e:
                logger.error(f"Error processing frame {frame_index}: {e}")
                continue  # Skip to the next frame if an error occurs
    
    except Exception as e:
        logger.error(f"Error in frame_to_npz {frame_index}: {e}")
        return

    try:   
        npz_extraction_complete.set() 
    except Exception as e:
        logger.error(f"Error processing frame {frame_index}: {e}")