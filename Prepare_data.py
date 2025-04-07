import os
import numpy as np
import shutil

def filter_kitti_labels_inplace(folder):
    allowed_classes = {"Car", "Van", "Pedestrian", "Cyclist"}
    max_distance = 20.0

    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue
        
        file_path = os.path.join(folder, filename)
        
        with open(file_path, 'r') as infile:
            lines = infile.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            label = parts[0]
            z_distance = float(parts[13])

            if label == "Van":
                label = "Car"
            
            if label in allowed_classes and z_distance <= max_distance:
                parts[0] = label
                new_lines.append(" ".join(parts) + '\n')
        
        with open(file_path, 'w') as outfile:
            outfile.writelines(new_lines)


def delete_empty_txt_and_related(txt_folder, lidar_folder, image_folder):
    total_deleted = 0
    for filename in os.listdir(txt_folder):
        if filename.endswith(".txt"):
            txt_path = os.path.join(txt_folder, filename)
            
            if os.path.getsize(txt_path) == 0:
                base_name = os.path.splitext(filename)[0]
                
                related_files = [
                    txt_path,
                    os.path.join(lidar_folder, base_name + ".bin"),
                    os.path.join(image_folder, base_name + ".png"),
                    os.path.join(calib_folder, base_name + ".txt")
                ]
                
                for file in related_files:
                    if os.path.exists(file):
                        os.remove(file)
                        total_deleted += 1
                        print(f"Verwijderd: {file}")


def fix_file_numbering(folder_path, extension):
    files = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    
    files.sort(key=lambda f: int(f.split('.')[0]))
    
    for i, file in enumerate(files):
        new_name = f"{i:06d}.{extension}"
        old_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_name)
        
        os.rename(old_file_path, new_file_path)

def filter_lidar_points(lidar_folder, max_x=21.0):
    for filename in os.listdir(lidar_folder):
        if filename.endswith(".bin"):
            file_path = os.path.join(lidar_folder, filename)
            
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            
            filtered_points = points[points[:, 0] <= max_x]
            
            filtered_points.tofile(file_path)
            print(f"Gefilterd: {file_path}, verwijderde punten: {len(points) - len(filtered_points)}")


label_folder = os.path.expanduser("~/main_folder/kitti/training/label_2")
lidar_folder = os.path.expanduser("~/main_folder/kitti/training/velodyne")
calib_folder = os.path.expanduser("~/main_folder/kitti/training/calib")
image_folder = os.path.expanduser("~/main_folder/kitti/training/image_2")

filter_kitti_labels_inplace(label_folder)
fix_file_numbering(calib_folder, "txt")
fix_file_numbering(image_folder, "png")
fix_file_numbering(lidar_folder, "bin")
fix_file_numbering(label_folder, "txt")
filter_lidar_points(lidar_folder)

total_frames = sum(1 for item in os.listdir(folder) if os.path.isfile(os.path.join(label_folder, item)))

all_frames = [f"{i:06d}" for i in range(total_frames)]

np.random.seed(42)
np.random.shuffle(all_frames)

train_split = int(0.7 * total_frames)
val_split = int(0.85 * total_frames)

train_set = sorted(all_frames[:train_split])
val_set = sorted(all_frames[train_split:val_split])
test_set = sorted(all_frames[val_split:])
trainval_set = sorted(train_set + val_set)

def save_set(filename, data):
    with open(filename, "w") as f:
        f.write("\n".join(data) + "\n")

save_set("train.txt", train_set)
save_set("val.txt", val_set)
save_set("test.txt", test_set)
save_set("trainval.txt", trainval_set)


base_dir = os.path.expanduser("~/kitti/training")
test_dir = os.path.expanduser("~/kitti/testing")

folders = {
    "calib": ".txt",
    "image_2": ".png",
    "label_2": ".txt",
    "velodyne": ".bin",
}

with open("test.txt", "r") as f:
    test_files = [line.strip() for line in f.readlines()]

for folder in folders.keys():
    os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

for file_num in test_files:
    for folder, ext in folders.items():
        src_path = os.path.join(base_dir, folder, f"{file_num}{ext}")
        dst_path = os.path.join(test_dir, folder, f"{file_num}{ext}")
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)