import subprocess

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
