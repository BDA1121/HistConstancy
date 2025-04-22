import os
import shutil

# List of source folders
source_folders = [
    '/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images_depth/folder_1',
    '/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images_depth/folder_2',
    '/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images_depth/folder_3',
    '/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images_depth/folder_4',
    '/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images_depth/folder_5',
    '/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images_depth/folder_6',
    '/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images_depth/folder_7',
    '/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images_depth/folder_8',
    '/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images_depth/folder_9'
]

# Destination folder where you want to move the images
destination_folder = '/work/SuperResolutionData/spectralRatio/data/images_for_training/depth/'

# Image extensions to look for
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

# Make sure destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate over each source folder
for folder in source_folders:
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            # Check if file is an image (based on extension)
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
                try:
                    # Move the file to the destination folder
                    shutil.move(file_path, destination_folder)
                    print(f"Moved: {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
    else:
        print(f"Folder not found: {folder}")

print("All images moved successfully!")
