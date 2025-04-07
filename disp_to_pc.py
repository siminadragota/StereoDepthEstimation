import os
import numpy as np
import cv2

def extract_matrices(filename):
    """
    Reads calibration data from a file and stores important matrices as global variables.
    
    Expected keys:
      - P0, P1, P2, P3 (3x4 projection matrices)
      - R0_rect (3x3 matrix)
      - Tr_velo_to_cam (3x4, extended to 4x4) and its inverse Tr_cam_to_velo
      - Tr_imu_to_velo (3x4, extended to 4x4) and its inverse Tr_velo_to_imu
    """
    global P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_cam_to_velo, Tr_imu_to_velo, Tr_velo_to_imu

    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                parts = line.split(':')
                key = parts[0].strip()
                values = np.fromstring(parts[1], sep=' ')

                if key.startswith("P"):
                    matrix = values.reshape(3, 4)
                elif key == "R0_rect":
                    matrix = values.reshape(3, 3)
                elif key in ["Tr_velo_to_cam", "Tr_imu_to_velo"]:
                    matrix_3x4 = values.reshape(3, 4)
                    bottom_row = np.array([[0, 0, 0, 1]])
                    matrix = np.vstack([matrix_3x4, bottom_row])
                    
                    # Calculate the inverse matrix and store it
                    inverse_matrix = np.linalg.inv(matrix)
                    if key == "Tr_velo_to_cam":
                        globals()["Tr_cam_to_velo"] = inverse_matrix
                    elif key == "Tr_imu_to_velo":
                        globals()["Tr_velo_to_imu"] = inverse_matrix
                else:
                    continue  # Unexpected key, skip

                # Save the matrix as a global variable
                globals()[key] = matrix

def transform_points(points, transform_matrix):
    """
    Transforms the points with the given transformation matrix.
    The points are first converted to homogeneous coordinates.
    """
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    transformed_points_homogeneous = points_homogeneous @ transform_matrix.T
    return transformed_points_homogeneous[:, :3]

def compute_3D_from_disparity(right_projection_matrix, disparity_img, R0_rect):
    """
    Calculate the 3D coordinates per pixel based on the disparity image, 
    with uniform subsampling.
    
    Parameters:
      - right_projection_matrix: The P3 matrix (3x4) of the camera
      - disparity_img: The disparity image as a float32 array
      - subsample_factor: The subsampling factor (e.g., 4 means 4x4 pixels per averaged pixel)
    
    Returns:
      - xyz_img: A (height x width x 3) array with the 3D position per pixel (or np.nan if not valid).
    """
    height, width = disparity_img.shape

    # Extract parameters from the projection matrix
    f = right_projection_matrix[0, 0]
    c_x = right_projection_matrix[0, 2]
    c_y = right_projection_matrix[1, 2]
    T_x = right_projection_matrix[0, 3]

    # Calculate depth (z) per pixel
    no_occlusions = (disparity_img > 0)
    depth_img = np.full_like(disparity_img, np.nan, dtype=np.float32)
    depth_img[no_occlusions] = (-T_x) / disparity_img[no_occlusions]

    # Create a meshgrid for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    X_cam = (u - c_x) * depth_img / f
    Y_cam = (v - c_y) * depth_img / f
    Z_cam = depth_img


    # Combine the subsampled X, Y, Z coordinates
    xyz_cam = np.stack((X_cam, Y_cam, Z_cam), axis=-1)

    # xyz_flat = xyz_cam.reshape(-1, 3)
    # xyz_rect = R0_rect @ xyz_flat.T
    # xyz_img = xyz_rect.reshape(height, width, 3)

    return xyz_cam

def count_files_in_directory(directory):
    """
    Returns the count of files in the given directory.
    """
    # Get all files in the directory (ignore subdirectories)
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)

def save_pointcloud_to_bin(points, filename):
    """
    Saves a point cloud to a binary file (.bin) in homogeneous coordinates.
    Each point will be stored as [x, y, z, 1].
    Input points should be a Nx3 or Nx4 array.
    """
    # Convert to numpy array if it isn't already
    points = np.asarray(points)
    
    # Check if points are already in homogeneous coordinates
    if points.shape[1] == 4:
        # Assume they're already homogeneous (though we might want to normalize)
        homogeneous_points = points
    else:
        # Convert to homogeneous coordinates by adding a column of 1s
        num_points = points.shape[0]
        ones = np.ones((num_points, 1), dtype=points.dtype)
        homogeneous_points = np.hstack((points, ones))
    
    print(homogeneous_points.shape)
    homogeneous_points.astype(np.float32).tofile(filename)


# Settings for file paths
file_path_dataset = os.path.expanduser("~/kitti")
calibration_relative_path = "/training/calib/{:06d}.txt"
pointcloud_dir = os.path.expanduser("~kitti/training/velodyne/")  # Directory to save pointclouds


# Get the number of disparity images
disparity_relative_path = "/training/disparity_images/{:06d}.png"  # Use the correct format for filenames

# Get the number of disparity images
disparity_dir = file_path_dataset + "/disparity_images/"
total_images = count_files_in_directory(disparity_dir)

# Process each image and save point clouds
for i in range(total_images):
    calibration_path = file_path_dataset + calibration_relative_path.format(i)
    disparity_path = file_path_dataset + disparity_relative_path.format(i)  # Correct file path for disparity image

    # Extract the calibration matrices
    extract_matrices(calibration_path)

    # Load the disparity image
    disparity_img = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    disparity_img /= 256.0  # Convert to meters (if required by dataset)

    # Compute the 3D point cloud
    xyz_img = compute_3D_from_disparity(P3, disparity_img, R0_rect)
    xyz_img_clean = np.nan_to_num(xyz_img)

    # Create a mask for valid points (non-NaN)
    valid_mask = ~np.isnan(xyz_img_clean).any(axis=2)
    points_3d = xyz_img_clean[valid_mask]

    # Transform points to lidar coordinate system
    points_3d_lidar = transform_points(points_3d, Tr_cam_to_velo)
    print(points_3d_lidar.shape)

    # Save the point cloud as a binary file
    pointcloud_filename = os.path.join(pointcloud_dir, f"{i:06d}.bin")
    save_pointcloud_to_bin(points_3d_lidar, pointcloud_filename)
    print(f"Saved point cloud {i:06d} as {pointcloud_filename}")
