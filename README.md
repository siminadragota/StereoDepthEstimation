
# PointPillars-Based Object Detection Using 3D Point Clouds from Stereo Disparity
This project is designed to train **PointPillars**, an efficient 3D object detection method, using point clouds that are generated from disparity images. The project involves several steps, including setting up the necessary repositories, generating disparity point clouds, preparing the data, and training the model.

## Setting Up the Environment

To set up the environment, use the following commands to create and activate the environment:

```shell
conda env create -f environment.yaml
conda activate cv_env
```

This will install all the necessary dependencies for the project.


## Visualizing Point Clouds and Predictions

You can visualize the point clouds by opening the following two Jupyter notebook files:

- **`PC_visualisation.ipynb`**: This notebook plots the two point clouds.
- **`detections_visualization.ipynb`**: This notebook displays the point clouds along with predictions and ground truth data.

Simply open these notebooks in Jupyter to explore the visualizations.


## Repositories and Directory Structure

To train **PointPillars** on point clouds generated from disparity images, you will need to clone several repositories. Your directory structure should look like this:

```
Main_folder/
├── StereoDepthEstimation/
├── PSMNET/
├── Pointpillars/
└── Kitti_dataset/
```

### Cloning and Setting Up the Repositories

#### PSMNET

1. Clone the [PSMNET GitHub repository](https://github.com/JiaRenChang/PSMNet) into the `PSMNET` folder in your project directory.
2. Install the required libraries as described in the PSMNET repository.
3. Follow the instructions in the repository to generate the disparity images from your input data.
4. Once generated, place the disparity images into the `Kitti_dataset` folder. These images will be used to generate the point clouds in a later step.

#### PointPillars

1. Clone the [PointPillars GitHub repository](https://github.com/zhulf0804/PointPillars) into the `Pointpillars` folder.
2. Install the necessary libraries as per the instructions in the PointPillars repository.

## Generating Disparity Point Clouds

Once the setup is complete, you can generate point clouds from the disparity images. Use the following command to generate the point clouds and transform them into the lidar frame:

```shell
python3 disp_to_pc.py
```

This step will create point clouds from the disparity images that can be fed into the PointPillars network for 3D object detection.

## Preparing the Data for PointPillars

The disparity point clouds have now been generated, but the entire dataset needs to be processed before it can be used for training with **PointPillars**. Run the following command to prepare the dataset. This process may take a few minutes as it involves copying and organizing files:

```shell
python3 Prepare_data.py
```

Once this step is completed, you will have all the necessary data ready for training the model.

## Training the PointPillars Network

The **PointPillars** network can now be trained on the processed point clouds. During the data preparation step, the `Prepare_data.py` script will generate four files:

- `test.txt`
- `train.txt`
- `trainval.txt`
- `val.txt`

These files need to be copied to the following directory within the **PointPillars** repository:

```
/Pointpillars/pointpillars/dataset/ImageSets/
```

Once the files are in place, you can begin training the PointPillars model by following the instructions on the [PointPillars GitHub page](https://github.com/zhulf0804/PointPillars).