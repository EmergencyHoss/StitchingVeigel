import os
from stitching import Stitcher
import cv2
import sys
from Utils import utils


def stitch_image_pairs(list1, list2, output_dir="stitched"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stitcher = Stitcher(detector="sift", confidence_threshold=0.2)

    for i, (img1_path, img2_path) in enumerate(zip(list1, list2)):
        try:
            print(f"Stitching pair {i+1}: {img1_path} + {img2_path}")
            result = stitcher.stitch([img1_path, img2_path])
            output_path = os.path.join(output_dir, f"stitched_{i+1}.jpg")
            cv2.imwrite(output_path, result)
        except Exception as e:
            print(f"Failed to stitch pair {i+1}: {e}")


list1 = utils.get_image_paths_from_folder(folder_path=r"data\images\3900001419")
list2 = utils.get_image_paths_from_folder(folder_path=r"data\images\3900001421")

stitch_image_pairs(list1, list2)

