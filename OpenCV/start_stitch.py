from stitching import Stitcher
import cv2



stitcher = Stitcher()
stitcher = Stitcher(detector="sift", confidence_threshold=0.3)
panorama = stitcher.stitch(["./resources/test_1.jpg", "./resources/test_2.jpg"])

cv2.imwrite("panorama.jpg", panorama)
