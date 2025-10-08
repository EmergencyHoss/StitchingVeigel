import pandas as pd
import logging
import cv2
import os

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
