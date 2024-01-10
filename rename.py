import os

root_dir = 'archive'

# # Traverse the directory structure
# for dirpath, dirnames, filenames in os.walk(root_dir):
#     for filename in filenames:
#         file_name, file_extension = os.path.splitext(filename)
#         if not file_extension:
#             old_path = os.path.join(dirpath, filename)
#             new_name = f"{file_name}.jpeg"  # New extension, you can modify this
#             new_path = os.path.join(dirpath, new_name)
#             os.rename(old_path, new_path)
#             print(f"Renamed {old_path} to {new_path}")


# Traverse the directory structure and rename directories
idx = 0
for dir_name in os.listdir(root_dir):  # Renaming from 0 to 43
    old_dir = os.path.join(root_dir, dir_name)
    new_dir = os.path.join(root_dir, f'class_{idx}')  # New directory name pattern
    idx += 1
    if os.path.exists(old_dir):
        os.rename(old_dir, new_dir)
        print(f"Renamed {old_dir} to {new_dir}")