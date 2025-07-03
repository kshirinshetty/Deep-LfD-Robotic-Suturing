import pandas as pd
import numpy as np
import os

def create_new_cs_ns_csv():
    """
    Create new CS and NS CSV files with fps/3 reduction
    Original: 59,406 entries -> New: 13,560 entries
    Train/Test split: 10,000 / 3,560
    """
    
    # Define full paths
    base_path = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data'
    
    # Original file paths
    original_cs_path = os.path.join(base_path, 'Arm2_CS_new.csv')
    original_ns_path = os.path.join(base_path, 'Arm2_NS_new.csv')
    
    # New file paths for fps/3 reduced data
    new_cs_path = os.path.join(base_path, 'Arm2_CS_fps3.csv')
    new_ns_path = os.path.join(base_path, 'Arm2_NS_fps3.csv')
    
    # Train/Test split file paths
    train_cs_path = os.path.join(base_path, 'Arm2_CS_train.csv')
    train_ns_path = os.path.join(base_path, 'Arm2_NS_train.csv')
    test_cs_path = os.path.join(base_path, 'Arm2_CS_test.csv')
    test_ns_path = os.path.join(base_path, 'Arm2_NS_test.csv')
    
    print("=== Creating New CS NS CSV Files ===")
    print(f"Base path: {base_path}")
    
    # Check if original files exist
    if not os.path.exists(original_cs_path):
        print(f"❌ Original CS file not found: {original_cs_path}")
        return
    
    if not os.path.exists(original_ns_path):
        print(f"❌ Original NS file not found: {original_ns_path}")
        return
    
    # Load original robot data
    print("\n📁 Loading original robot data...")
    Arm2_CS_State = pd.read_csv(original_cs_path, header=None)
    Arm2_NS_State = pd.read_csv(original_ns_path, header=None)
    
    print(f"Original robot state shapes - CS: {Arm2_CS_State.shape}, NS: {Arm2_NS_State.shape}")
    
    # Apply fps/3 reduction - take every 3rd entry
    print("\n🔄 Applying fps/3 reduction (every 3rd entry)...")
    filtered_cs = Arm2_CS_State.iloc[::3].reset_index(drop=True)
    filtered_ns = Arm2_NS_State.iloc[::3].reset_index(drop=True)
    
    print(f"After fps/3 reduction - CS: {filtered_cs.shape}, NS: {filtered_ns.shape}")
    
    # Verify we have the expected 13,560 entries
    expected_total = 13560
    actual_total = len(filtered_cs)
    
    if actual_total != expected_total:
        print(f"⚠️  Warning: Expected {expected_total} entries, got {actual_total}")
        if actual_total < expected_total:
            print("📊 Using available data size")
        else:
            print("✂️  Trimming to expected size")
            filtered_cs = filtered_cs[:expected_total]
            filtered_ns = filtered_ns[:expected_total]
    
    # Save fps/3 reduced files
    print(f"\n💾 Saving fps/3 reduced files...")
    filtered_cs.to_csv(new_cs_path, header=False, index=False)
    filtered_ns.to_csv(new_ns_path, header=False, index=False)
    print(f"✅ Saved: {new_cs_path}")
    print(f"✅ Saved: {new_ns_path}")
    
    # Split into train/test sets
    print(f"\n📊 Creating train/test splits...")
    train_size = 10000
    test_size = 3560
    total_needed = train_size + test_size
    
    # Verify we have enough data
    if len(filtered_cs) < total_needed:
        print(f"⚠️  Warning: Need {total_needed} samples but only have {len(filtered_cs)}")
        available = len(filtered_cs)
        train_size = min(10000, available - 1000)  # Leave some for test
        test_size = available - train_size
        print(f"📏 Adjusted - Train: {train_size}, Test: {test_size}")
    
    # Create train/test splits
    robot_state_train_input = filtered_cs[:train_size]
    robot_state_train_label = filtered_ns[:train_size]
    robot_state_test_input = filtered_cs[train_size:train_size + test_size]
    robot_state_test_label = filtered_ns[train_size:train_size + test_size]
    
    print(f"\n📈 Final split sizes:")
    print(f"Train CS: {robot_state_train_input.shape}")
    print(f"Train NS: {robot_state_train_label.shape}")
    print(f"Test CS: {robot_state_test_input.shape}")
    print(f"Test NS: {robot_state_test_label.shape}")
    
    # Save train/test splits
    print(f"\n💾 Saving train/test split files...")
    robot_state_train_input.to_csv(train_cs_path, header=False, index=False)
    robot_state_train_label.to_csv(train_ns_path, header=False, index=False)
    robot_state_test_input.to_csv(test_cs_path, header=False, index=False)
    robot_state_test_label.to_csv(test_ns_path, header=False, index=False)
    
    print(f"✅ Saved train CS: {train_cs_path}")
    print(f"✅ Saved train NS: {train_ns_path}")
    print(f"✅ Saved test CS: {test_cs_path}")
    print(f"✅ Saved test NS: {test_ns_path}")
    
    # Summary report
    print(f"\n📋 === SUMMARY REPORT ===")
    print(f"📁 Base directory: {base_path}")
    print(f"📊 Original data size: {Arm2_CS_State.shape[0]} entries")
    print(f"🔄 After fps/3 reduction: {len(filtered_cs)} entries")
    print(f"🎯 Train/Test split: {train_size} / {test_size}")
    print(f"📝 Created files:")
    print(f"   - {os.path.basename(new_cs_path)}")
    print(f"   - {os.path.basename(new_ns_path)}")
    print(f"   - {os.path.basename(train_cs_path)}")
    print(f"   - {os.path.basename(train_ns_path)}")
    print(f"   - {os.path.basename(test_cs_path)}")
    print(f"   - {os.path.basename(test_ns_path)}")
    print(f"✅ All files created successfully!")
    
    # Load filtered image filenames to verify alignment
    filtered_dir = os.path.join(base_path, 'filtered')
    train_filtered_path = os.path.join(filtered_dir, 'train_filtered.csv')
    test_filtered_path = os.path.join(filtered_dir, 'test_filtered.csv')
    
    if os.path.exists(train_filtered_path) and os.path.exists(test_filtered_path):
        print(f"\n🔍 Verifying alignment with filtered images...")
        X_train_filenames = pd.read_csv(train_filtered_path, header=None)
        X_test_filenames = pd.read_csv(test_filtered_path, header=None)
        
        print(f"📸 Train images: {len(X_train_filenames)}")
        print(f"📸 Test images: {len(X_test_filenames)}")
        print(f"🤖 Train robot states: {len(robot_state_train_input)}")
        print(f"🤖 Test robot states: {len(robot_state_test_input)}")
        
        if (len(X_train_filenames) == len(robot_state_train_input) and 
            len(X_test_filenames) == len(robot_state_test_input)):
            print("✅ Perfect alignment! Images and robot states match.")
        else:
            print("❌ Misalignment detected between images and robot states.")
            print("💡 You may need to adjust the data splits to match.")
    else:
        print(f"\n⚠️  Filtered image files not found:")
        print(f"   - {train_filtered_path}")
        print(f"   - {test_filtered_path}")
        print(f"💡 Make sure to run your filtering script first.")

def verify_data_integrity():
    """
    Verify the integrity of the created CSV files
    """
    base_path = '/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/data/Data'
    
    files_to_check = [
        'Arm2_CS_fps3.csv',
        'Arm2_NS_fps3.csv',
        'Arm2_CS_train.csv',
        'Arm2_NS_train.csv',
        'Arm2_CS_test.csv',
        'Arm2_NS_test.csv'
    ]
    
    print("\n🔍 === DATA INTEGRITY CHECK ===")
    
    for filename in files_to_check:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, header=None)
            print(f"✅ {filename}: {df.shape} - {df.dtypes.nunique()} data types")
            
            # Check for missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                print(f"   ⚠️  Warning: {missing_count} missing values found")
            
            # Check data range
            print(f"   📊 Value range: {df.min().min():.4f} to {df.max().max():.4f}")
        else:
            print(f"❌ {filename}: File not found")

if __name__ == "__main__":
    # Create the new CSV files
    create_new_cs_ns_csv()
    
    # Verify data integrity
    verify_data_integrity()
    
    print(f"\n🎉 Script completed successfully!")
    print(f"💡 Next steps:")
    print(f"   1. Update your training script to use the new CSV files")
    print(f"   2. Verify alignment with your filtered image files")
    print(f"   3. Run a quick test to ensure data loads correctly")