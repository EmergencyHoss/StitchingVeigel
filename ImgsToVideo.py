# Now you can import your module
from src.Utils import utils

utils.create_video_from_images(
    image_folder=r".\data\images\panoramas",
    framerate=60,
    output_dir=r".\data\videos"
)
