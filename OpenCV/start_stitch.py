from stitching import Stitcher
import cv2

data_path = "./data/images/"
stitcher = Stitcher()
stitcher = Stitcher(detector="sift", confidence_threshold=0.2)

max = 10
zerosPath = "0000"

import numpy as np

def cylindrical_projection(img, focal_length):
    h, w = img.shape[:2]
    # Create output image
    cyl = np.zeros_like(img)
    center_x = w // 2
    center_y = h // 2

    for y in range(h):
        for x in range(w):
            # Convert to cylindrical coordinates
            theta = np.arctan((x - center_x) / focal_length)
            h_ = (y - center_y) / np.sqrt((x - center_x)**2 + focal_length**2)
            x_cyl = int(focal_length * theta + center_x)
            y_cyl = int(focal_length * h_ + center_y)
            if 0 <= x_cyl < w and 0 <= y_cyl < h:
                cyl[y_cyl, x_cyl] = img[y, x]
    return cyl

for i in range(1, 400) :
    try:
        panorama = stitcher.stitch([f"{data_path}3900001421/frame_{zerosPath}{i-1}.jpg", f"{data_path}3900001419/frame_{zerosPath}{i-1}.jpg"])
        if(i == 10 or i == 100):
            max*=10
            zerosPath = zerosPath[:-1]
        
        cv2.imwrite(f"{data_path}/panoramas/panorama{i-1}.jpg", panorama)
    except Exception as e:
        print(f"Error at image {i-1}: {e}")
        continue

img = cv2.imread(f"{data_path}/panoramas/panorama0.jpg")
focal_length = 500  # Adjust as needed
cyl_img = cylindrical_projection(img, focal_length)
cv2.imwrite('cylindrical_panorama.jpg', cyl_img)
