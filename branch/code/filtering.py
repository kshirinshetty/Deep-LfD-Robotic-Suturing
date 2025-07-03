import os
import pandas as pd

# Paths
csv_dir = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data'
images_dir = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results/all_images'
output_dir = os.path.join(csv_dir, 'filtered')
os.makedirs(output_dir, exist_ok=True)

# Step 1: Read all image names from both CSVs
train_csv = pd.read_csv(os.path.join(csv_dir, 'trainImageName.csv'), header=None).values[:, 0]
test_csv = pd.read_csv(os.path.join(csv_dir, 'testImageName.csv'), header=None).values[:, 0]

# Step 2: Combine and deduplicate (if needed)
all_names = list(dict.fromkeys(list(train_csv) + list(test_csv)))

# Step 3: Filter names that exist in the all_images directory
existing_names = [
    name for name in all_names if os.path.exists(os.path.join(images_dir, name))
]

# Check count
if len(existing_names) != 13560:
    raise ValueError(f"Expected 13,560 matching images, found {len(existing_names)}.")

# Step 4: Split
train_names = existing_names[:10000]
test_names = existing_names[10000:]

# Step 5: Save
pd.DataFrame(train_names).to_csv(os.path.join(output_dir, 'train_filtered.csv'), index=False, header=False)
pd.DataFrame(test_names).to_csv(os.path.join(output_dir, 'test_filtered.csv'), index=False, header=False)

print("✅ Done: Saved 10,000 train and 3,560 test image names.")
