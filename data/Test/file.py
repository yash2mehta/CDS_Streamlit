import os
import shutil

# Function to rename files in a folder and copy them to a destination folder
def rename_and_copy_files(src_folder, dst_folder, prefix=None):
    files = os.listdir(src_folder)
    for filename in files:
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        if prefix:
            # Rename file
            name, extension = os.path.splitext(filename)
            new_filename = f"{prefix}_{name}"
            dst_path = os.path.join(dst_folder, new_filename + extension)
        shutil.copyfile(src_path, dst_path)

# Folder paths
melanoma_folder = "melanoma"
no_melanoma_folder = "no_melanoma"
test_all_folder = "Test_All"

# Create Test_All directory if it doesn't exist
if not os.path.exists(test_all_folder):
    os.makedirs(test_all_folder)

# Rename and copy files from melanoma folder to Test_All
rename_and_copy_files(melanoma_folder, test_all_folder, prefix="1")

# Copy files from no_melanoma folder to Test_All
rename_and_copy_files(no_melanoma_folder, test_all_folder, prefix="0")

print("Files copied successfully!")