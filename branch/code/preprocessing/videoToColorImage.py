import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
import glob
import sys
sys.setrecursionlimit(10000)

# Updated file paths
base_path = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data'
results_path = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results'
all_images_dir = os.path.join(results_path, 'all_images')

print("Loading images from existing directory...")
print(f"Images directory: {all_images_dir}")

# Check if directory exists
if not os.path.exists(all_images_dir):
    print(f"Error: Directory {all_images_dir} does not exist!")
    sys.exit(1)

# Get all image files
image_files = glob.glob(os.path.join(all_images_dir, "*.jpg"))
print(f"Found {len(image_files)} image files")

# Sort files to ensure consistent ordering
# Expected format: T{testNumber}Cam{vidNameCounter}_img{count}.jpg
def sort_key(filename):
    basename = os.path.basename(filename)
    # Extract test number, camera, and image number for proper sorting
    try:
        # Format: T1Cam1_img1.jpg -> extract 1, 1, 1
        parts = basename.replace('.jpg', '').split('_')
        test_cam = parts[0]  # T1Cam1
        img_num = int(parts[1].replace('img', ''))  # 1
        
        test_num = int(test_cam.split('Cam')[0].replace('T', ''))  # 1
        cam_num = int(test_cam.split('Cam')[1])  # 1
        
        return (test_num, cam_num, img_num)
    except:
        return (0, 0, 0)

image_files.sort(key=sort_key)
print(f"Sorted {len(image_files)} image files")

# Load images into array
print("Loading images into numpy array...")
all_images = []
failed_count = 0

for i, img_file in enumerate(image_files):
    try:
        # Load image using PIL to ensure RGB format
        image = Image.open(img_file)
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Ensure image is 224x224 (should already be, but verify)
        if image.size != (224, 224):
            image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.uint8)
        all_images.append(image_array)
        
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i + 1}/{len(image_files)} images...")
            
    except Exception as e:
        print(f"Error loading {img_file}: {e}")
        failed_count += 1
        continue

print(f"Successfully loaded {len(all_images)} images")
if failed_count > 0:
    print(f"Failed to load {failed_count} images")

# Convert to numpy array
if all_images:
    print("Converting to numpy array...")
    x_array = np.array(all_images, dtype=np.uint8)
    print(f"Final dataset shape: {x_array.shape}")
    
    # Save the array
    save_path = os.path.join(results_path, 'Test1-60_fps3_images.npy')
    np.save(save_path, x_array)
    print(f"Dataset saved to: {save_path}")
    
    # Print some statistics
    print(f"Data type: {x_array.dtype}")
    print(f"Memory usage: {x_array.nbytes / (1024*1024):.2f} MB")
    print(f"Image shape: {x_array.shape[1:]}")
    print(f"Pixel value range: [{x_array.min()}, {x_array.max()}]")
    
    # Verify the target count
    if len(all_images) == 13560:
        print("✅ Successfully created array with exactly 13,560 images!")
    else:
        print(f"⚠️  Expected 13,560 images but got {len(all_images)}")
        
    # Show first few filenames for verification
    print("\nFirst 10 image files (for verification):")
    for i in range(min(10, len(image_files))):
        print(f"  {os.path.basename(image_files[i])}")
        
else:
    print("❌ No images were loaded successfully!")

print("Array creation from existing images completed.")