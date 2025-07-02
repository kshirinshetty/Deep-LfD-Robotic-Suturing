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
base_path = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data'
results_path = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results'

fname = os.path.join(base_path, 'DatasetSize.csv')
testSize = pd.read_csv(fname, header=None)
print("Dataset sizes:", testSize)

videoList = ['p1_left.mp4', 'p1_right.mp4', 'p2_left.mp4', 'p2_right.mp4', 'p3_left.mp4', 'p3_right.mp4']
cameraName = ['p1_left', 'p1_right', 'p2_left', 'p2_right', 'p3_left', 'p3_right']

# Process specific test range (modify as needed)
test_start = 1
test_end = 21  # Processing tests 1-20

for testNumber in range(test_start, test_end):
    test_path = os.path.join(base_path, str(testNumber))
    print(f"Processing test {testNumber}: {test_path}")
    
    if not os.path.exists(test_path):
        print(f"Warning: Path {test_path} does not exist, skipping...")
        continue
    
    os.chdir(test_path)
    
    for vidNameCounter in range(1, 7):
        camera_name = cameraName[vidNameCounter - 1]
        video_file = videoList[vidNameCounter - 1]
        
        if not os.path.exists(video_file):
            print(f"Warning: Video file {video_file} not found")
            continue
        
        # Create camera-specific directory
        camera_dir = f'results/camera_{camera_name}_data'
        os.makedirs(camera_dir, exist_ok=True)
        
        print(f"Processing camera {vidNameCounter}: {camera_name}")
        vidcap = cv2.VideoCapture(video_file)

        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = vidcap.read()
            if hasFrames:
                # Resize image to 224x224 for consistency with neural network input
                image_resized = cv2.resize(image, (224, 224))
                img_path = os.path.join(camera_dir, f"{count}.jpg")
                cv2.imwrite(img_path, image_resized)
                return True
            return False

        # Get video properties
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        print(f"FPS: {fps}, Duration: {duration:.2f}s")
        
        # Extract frames
        sec = 5.05  # Start time for image capture
        
        # Fix indexing - DatasetSize.csv might have data in columns, not rows
        if testSize.shape[0] == 1:  # If only one row, data is in columns
            num_frames = int(testSize.iloc[0, testNumber-1])  # Get from column
        else:  # If multiple rows, data is in rows
            num_frames = int(testSize.iloc[testNumber-1, 0])  # Get from row
            
        frameRate = (duration - sec) / num_frames if num_frames > 0 else 1.0
        
        count = 1
        success = getFrame(sec)
        while success and count < num_frames:
            count += 1
            sec = sec + frameRate
            sec = round(sec, 4)
            success = getFrame(sec)
        
        vidcap.release()

# Create consolidated dataset from extracted images
print("Creating consolidated image dataset...")

testSize_array = np.array(testSize)
subFolder = ['camera_p1_left_data', 'camera_p1_right_data', 'camera_p2_left_data', 
             'camera_p2_right_data', 'camera_p3_left_data', 'camera_p3_right_data']

# Calculate total number of images
total_images = 0
for i in range(test_start-1, test_end-1):
    if testSize.shape[0] == 1:  # Data in columns
        total_images += int(testSize.iloc[0, i]) * 6  # 6 cameras per test
    else:  # Data in rows
        total_images += int(testSize.iloc[i, 0]) * 6  # 6 cameras per test

print(f"Expected total images: {total_images}")

# Load and preprocess images
x = []
os.chdir(base_path)

for i in range(test_start, test_end):
    test_dir = os.path.join(base_path, str(i))
    if not os.path.exists(test_dir):
        continue
    
    # Fix indexing for num_frames
    if testSize.shape[0] == 1:  # Data in columns
        num_frames = int(testSize.iloc[0, i-1])
    else:  # Data in rows
        num_frames = int(testSize.iloc[i-1, 0])
    
    for j in range(1, num_frames + 1):
        for folder in subFolder:
            camera_path = os.path.join(test_dir, folder)
            if not os.path.exists(camera_path):
                continue
                
            img_file = os.path.join(camera_path, f"{j}.jpg")
            if os.path.exists(img_file):
                try:
                    image = Image.open(img_file)
                    # Ensure image is RGB and resize to 224x224
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image_resized = image.resize((224, 224))
                    image_array = np.array(image_resized)
                    x.append(image_array)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")

print(f"Loaded {len(x)} images")

# Save preprocessed dataset
if x:
    x_array = np.array(x)
    print(f"Final dataset shape: {x_array.shape}")
    
    save_path = os.path.join(results_path, f'Test{test_start}-{test_end-1}_images.npy')
    np.save(save_path, x_array)
    print(f"Dataset saved to: {save_path}")
else:
    print("No images were loaded successfully!")

print("Video to image conversion completed.")
