import pandas as pd
import logging
import cv2
import os
import re
from typing import Tuple, Optional
from typing import List

logger = logging.getLogger(__name__)

def read_parquet_file(path: str) -> pd.DataFrame:
    """
    Reads a Parquet file and returns a pandas DataFrame.

    Args:
        path (str): Path to the Parquet file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_parquet(path)
        logger.info(f"Successfully read {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return pd.DataFrame()
    

def extract_frames_from_video(video_path: str, output_dir: str = None) -> None:
    """
    Extracts frames from a video file and saves them as images in a folder named after the video file.

    Args:
        video_path (str): Path to the video file.
        output_dir (str, optional): Directory where the folder will be created. Defaults to the video's directory.
    """
    # Get the base name of the video file (without extension)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Determine the output directory
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    frame_folder = os.path.join(output_dir, video_name)

    # Create the folder if it doesn't exist
    os.makedirs(frame_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Save frame as image
        frame_filename = os.path.join(frame_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

def read_image(image_path: str) -> any:
    """
    Reads an image from the given path and returns it as a NumPy array.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
    return image

def read_video(video_path: str) -> cv2.VideoCapture:
    """
    Opens a video file and returns the VideoCapture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None
    return cap

def natural_sort_key(s: str) -> list:
    """
    Generates a key for natural sorting of strings with embedded numbers.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def create_video_from_images(image_folder: str, output_dir: str, output_format: str = ".mp4",
                             resolution: Optional[Tuple[int, int]] = None, framerate: int = 20) -> None:
    """
    Creates a video from images in a specified folder. The output video filename is derived from the image folder name.

    Args:
        image_folder (str): Path to the folder containing images.
        output_dir (str): Directory to save the output video file.
        output_format (str): Video file format (e.g., '.mp4', '.avi').
        resolution (Tuple[int, int], optional): Resolution of the output video (width, height). If None, uses image resolution.
        framerate (int): Frames per second for the output video.
    """
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))
    ]

    image_files.sort(key=natural_sort_key)

    if not image_files:
        print("No image files found in the folder.")
        return

    folder_name = os.path.basename(os.path.normpath(image_folder))
    output_filename = f"{folder_name}{output_format}"
    output_path = os.path.join(output_dir, output_filename)

    # Read first image to determine resolution if not provided
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Error: Could not read the first image {image_files[0]}")
        return

    if resolution is None:
        resolution = (first_image.shape[1], first_image.shape[0])  # (width, height)

    # Choose codec based on format
    codec_map = {
        ".mp4": 'mp4v',
        ".avi": 'XVID',
        ".mov": 'MJPG',
        ".mkv": 'X264'
    }
    fourcc = cv2.VideoWriter_fourcc(*codec_map.get(output_format.lower(), 'mp4v'))

    video_writer = cv2.VideoWriter(output_path, fourcc, framerate, resolution)

    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print(f"Warning: Could not read image {image_file}")
            continue
        if resolution != (img.shape[1], img.shape[0]):
            img = cv2.resize(img, resolution)
        video_writer.write(img)

    video_writer.release()
    print(f"✅ Video saved to: {output_path}")


   
def get_image_paths_from_folder(folder_path: str, extensions: List[str] = ['.bmp', '.jpg', '.jpeg', '.png']) -> List[str]:
    """
    Returns a sorted list of image file paths from a folder.

    Args:
        folder_path (str): Path to the folder containing images.
        extensions (List[str]): List of allowed image file extensions.

    Returns:
        List[str]: Sorted list of image file paths.
    """
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if any(f.lower().endswith(ext) for ext in extensions)
    ]
    return sorted(image_paths)
