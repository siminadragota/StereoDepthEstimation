import os

# Path to the file containing the list of numbers
input_file = '/home/simina/Computer_vision/Data_splitting/test.txt'
# Path to the folder containing .txt files to check against
folder_path = '/home/simina/Computer_vision/Performance/Test_set/disparity/submit'  # <- Replace this with the actual path
# folder_path = '/home/simina/Computer_vision/Performance/Evaluation_set/lidar/submit'

# Read allowed filenames from the file (without .txt extension)
with open(input_file, 'r') as f:
    valid_files = {line.strip() for line in f if line.strip()}

# Loop through files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        base_name = os.path.splitext(file_name)[0]
        if base_name not in valid_files:
            # File is not in the list, delete it
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            print(f"Deleted: {file_name}")

print("Cleanup complete.")

