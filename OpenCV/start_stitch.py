from stitching import Stitcher
import cv2

data_path = "./data/images/episode_001000_"
stitcher = Stitcher()
stitcher = Stitcher(detector="sift", confidence_threshold=0.01)

stitcher.settings["nfeatures"]=4000

maxZeros = 10
zerosPath = "0000"

for i in range(1, 9) :
    try:
        rf = f"{data_path}rf/frame_{zerosPath}{i-1}.png"
        rb = f"{data_path}rb/frame_{zerosPath}{i-1}.png"

        #panorama = stitcher.stitch([f"{data_path}3900001421/frame_{zerosPath}{i-1}.jpg", f"{data_path}3900001419/frame_{zerosPath}{i-1}.jpg"])

        panorama = stitcher.stitch([
            #f"{data_path}fl/frame_{zerosPath}{i-1}.jpg", 
            rf, rb, 
            #f"{data_path}r/frame_{zerosPath}{i-1}.jpg", 
            #f"{data_path}lb/frame_{zerosPath}{i-1}.jpg", 
            #f"{data_path}lf/frame_{zerosPath}{i-1}.jpg"
        ])

        if(i == maxZeros):
            maxZeros*=10
            zerosPath = zerosPath[:-1]
        
        cv2.imwrite(f"{data_path}/panoramas/panorama{i-1}.jpg", panorama)
    except Exception as e:
        print(f"Error at image {i-1}: {e}")
        continue
