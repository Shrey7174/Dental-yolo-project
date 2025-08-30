import os
import random
import shutil

def split_dataset(base_dir, dest_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Splits a dataset of images and YOLO labels into train, validation, and test sets.
    This version reads from separate 'images' and 'labels' subdirectories.

    Args:
        base_dir (str): Path to the 'dataset' directory containing 'images' and 'labels' folders.
        dest_dir (str): Path to the root directory where 'data/' will be created.
        train_ratio (float): Proportion of the dataset to allocate for training.
        val_ratio (float): Proportion of the dataset to allocate for validation.
    """
    source_images_dir = os.path.join(base_dir, 'images')
    source_labels_dir = os.path.join(base_dir, 'labels')

    if not os.path.isdir(source_images_dir) or not os.path.isdir(source_labels_dir):
        print(f"Error: Ensure '{base_dir}' contains 'images' and 'labels' subdirectories.")
        return

    all_images = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png'))]
    
    valid_files = []
    for img_file in all_images:
        basename = os.path.splitext(img_file)[0]
        label_file = f"{basename}.txt"
        if os.path.exists(os.path.join(source_labels_dir, label_file)):
            valid_files.append(img_file)
        else:
            print(f"Warning: Skipping image '{img_file}' as it has no matching label.")

    print(f"Found {len(all_images)} images and {len(os.listdir(source_labels_dir))} labels.")
    print(f"Processing {len(valid_files)} valid image-label pairs.")

    random.shuffle(valid_files)

    total_files = len(valid_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    train_files = valid_files[:train_end]
    val_files = valid_files[train_end:val_end]
    test_files = valid_files[val_end:]

    print(f"\nSplitting into:")
    print(f"  Training set:   {len(train_files)} images")
    print(f"  Validation set: {len(val_files)} images")
    print(f"  Test set:       {len(test_files)} images")

    sets = {'train': train_files, 'val': val_files, 'test': test_files}
    for set_name in sets:
        os.makedirs(os.path.join(dest_dir, 'data', 'images', set_name), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'data', 'labels', set_name), exist_ok=True)

    for set_name, file_list in sets.items():
        for filename in file_list:
            basename = os.path.splitext(filename)[0]
            
            image_src = os.path.join(source_images_dir, filename)
            label_src = os.path.join(source_labels_dir, f"{basename}.txt")

            image_dest = os.path.join(dest_dir, 'data', 'images', set_name, filename)
            label_dest = os.path.join(dest_dir, 'data', 'labels', set_name, f"{basename}.txt")

            shutil.copy2(image_src, image_dest)
            shutil.copy2(label_src, label_dest)

    print("\nDataset split successfully!")

if __name__ == '__main__':
    
    DATASET_DIR = 'dataset' 
    PROJECT_ROOT = '.' 
    
    split_dataset(DATASET_DIR, PROJECT_ROOT)