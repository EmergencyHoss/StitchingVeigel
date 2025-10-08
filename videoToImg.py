# Now you can import your module
from src.Utils import utils

utils.extract_frames_from_video(
    video_path=r".\data\videos\3900001421.mp4",
    
    output_dir=r".\data\images"
)