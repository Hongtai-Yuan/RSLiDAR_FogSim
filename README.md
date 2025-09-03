# RSLidar_Fogsim

## Overview

    .
    ├── file_lists                          # contains file lists for pointcloud_viewer.py
    │   └── ...
    ├── integral_lookup_tables              # contains lookup tables to speed up the fog simulation
    │   └── ... 
    ├── extract_fog.py                      # to extract real fog noise* from the SeeingThroughFog dataset
    ├── dualradar_fog.py                    # to realtime extract real fog noise* from the RSLidar dataset
    ├── fog_simulation.py                   # to augment a clear weather pointcloud with artificial fog (used during training)
    ├── generate_integral_lookup_table.py   # to precompute the integral inside the fog equation
    ├── pointcloud_viewer.py                # to visualize entire point clouds of different datasets with the option to augment fog into their scenes
    ├── README.md
    └── theory.py                           # to visualize the theory behind a single LiDAR beam in foggy conditions

## Getting Started

### Setup

1) Install [anaconda](https://docs.anaconda.com/anaconda/install/).

2) Create a new conda environment.

```bash
conda create --name foggy_lidar python=3.9 -y
```

3) Activate the newly created conda environment.

```bash
conda activate foggy_lidar
```

4) Install all necessary packages.

```bash
conda install matplotlib numpy opencv pandas plyfile pyopengl pyqt pyqtgraph quaternion scipy tqdm -c conda-forge -y
pip install pyquaternion
```

5) Clone this repository (including submodules).
```bash
git clone https://github.com/Hongtai-Yuan/RSLidar_Fogsim.git
cd RSLidar_Fogsim
```

### Usage

How to run the script that visualizes the theory behind a single LiDAR beam in foggy conditions:

```bash
python theory.py
```
![theory](https://user-images.githubusercontent.com/14181188/115370049-f9b74200-a1c8-11eb-88d0-474b8dd5daa3.gif)

How to run the script that visualizes entire point clouds of different datasets:

```bash
python pointcloud_viewer.py -d <path_to_where_you_store_your_datasets>
```

### This repository is based on https://github.com/MartinHahner/LiDAR_fog_sim.git
