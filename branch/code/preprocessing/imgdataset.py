import cv2
import os
import numpy as np
import glob
import pandas as pd
from scipy import ndimage
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10000)

# Updated file paths
fname = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data/DatasetSize.csv'
testSize = pd.read_csv(fname, header=None)
print("Dataset sizes:", testSize)

start_imaging = np.array([3.7, 4.9, 4.8, 3.4, 4.2, 3.5, 6, 4.9, 3.9, 5.2, 4.5, 4.4, 3.3, 4.95, 4.5, 4.05, 5.95, 5.9, 3.85, 5.05, 6.4, 3.5, 6.5, 4.05, 5.2, 4.3, 3.85, 5.9, 6.1, 6.85, 5.5, 7.1, 3.9, 6, 3.1, 5.9, 4.9, 4.4, 3.8, 4.7, \
6.1, 5.2, 4.15, 3.4, 5.5, 5.6, 5.3, 3.4, 5.25, 2.9, 4.15, 4.95, 5.9, 4.8, 3.2, 4.4, 5.9, 5.4, 4.3, 4.5])

videoList = ['p1_left.mp4', 'p1_right.mp4', 'p2_left.mp4', 'p2_right.mp4', 'p3_left.mp4', 'p3_right.mp4']
cameraName = ['p1_left', 'p1_right', 'p2_left', 'p2_right', 'p3_left', 'p3_right']

cwd = os.getcwd()
print("Current working directory:", cwd)

# Create results directory structure
results_dir = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results'
all_images_dir = os.path.join(results_dir, 'all_images')
os.makedirs(all_images_dir, exist_ok=True)

imgNameList = []
base_path = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data'

for testNumber in range(1, 61):
    test_path = os.path.join(base_path, str(testNumber))
    print(f"Processing test {testNumber}: {test_path}")
    
    if not os.path.exists(test_path):
        print(f"Warning: Path {test_path} does not exist, skipping...")
        continue
    
    os.chdir(test_path)

    for vidNameCounter in range(1, 7):
        video_file = videoList[vidNameCounter - 1]
        
        if not os.path.exists(video_file):
            print(f"Warning: Video file {video_file} not found in {test_path}")
            continue
            
        print(f"Processing camera {vidNameCounter}: {video_file}")
        vidcap = cv2.VideoCapture(video_file)

        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = vidcap.read()
            if hasFrames:
                # Resize image to 224x224 for AlexNet/CNN compatibility
                image_resized = cv2.resize(image, (224, 224))
                imgName = f"T{testNumber}Cam{vidNameCounter}_img{count}.jpg"
                imgSave = os.path.join(all_images_dir, imgName)
                cv2.imwrite(imgSave, image_resized)
                imgNameList.append(imgName)
                return True
            return False

        # Get video properties
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Calculate frame extraction parameters
        sec = start_imaging[testNumber-1]  # Start time for image capture
        
        # Fix indexing - DatasetSize.csv might have data in columns, not rows
        if testSize.shape[0] == 1:  # If only one row, data is in columns
            num_frames = int(testSize.iloc[0, testNumber-1])  # Get from column
        else:  # If multiple rows, data is in rows
            num_frames = int(testSize.iloc[testNumber-1, 0])  # Get from row
            
        frameRate = (duration - sec) / num_frames if num_frames > 0 else 1.0
        
        print(f"Video duration: {duration:.2f}s, Start time: {sec}s, Frame rate: {frameRate:.4f}s")
        
        count = 1
        success = getFrame(sec)
        while success and count < num_frames:
            count += 1
            sec = sec + frameRate
            sec = round(sec, 4)
            success = getFrame(sec)
        
        vidcap.release()

# Save image name list for training/testing split
df = pd.DataFrame(imgNameList, columns=['image_name'])
print(f"Total images extracted: {df.shape[0]}")

# Save complete image list
image_list_path = os.path.join(results_dir, 'imageNameList.csv')
df.to_csv(image_list_path, header=False, index=False)

print(f"Image extraction completed. Image list saved to: {image_list_path}")
print(f"Images saved to: {all_images_dir}")
