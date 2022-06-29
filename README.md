# Intrinsic3D

## High-Quality 3D Reconstruction by Joint Appearance and Geometry Optimization with Spatially-Varying Lighting (ICCV 2017)

[![Intrinsic3D](https://vision.in.tum.de/_media/data/datasets/intrinsic3d/maier2017intrinsic3d_teaser.jpg?w=700&tok=b8e6f7)](https://vision.in.tum.de/_media/spezial/bib/maier2017intrinsic3d.pdf)

### License ###
Copyright (c) 2019, NVIDIA Corporation and Technical University of Munich. All Rights Reserved.
The Intrinsic3D source code is available under the [BSD license](http://opensource.org/licenses/BSD-3-Clause), please see the [LICENSE](LICENSE) file for details.
All data in the [Intrinsic3D Dataset](https://vision.in.tum.de/data/datasets/intrinsic3d) is licensed under a [Creative Commons 4.0 Attribution License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), unless stated otherwise.

### Resources ###
* [Project page @ NVIDIA Research](https://research.nvidia.com/publication/2017-10_Intrinsic3D%3A-High-Quality-3D)
* [Intrinsic3D Dataset @ TUM](https://vision.in.tum.de/data/datasets/intrinsic3d)
* [ICCV 2017 Paper (PDF)](https://vision.in.tum.de/_media/spezial/bib/maier2017intrinsic3d.pdf)
* [ArXiv (PDF)](https://arxiv.org/pdf/1708.01670.pdf)
 
### Project members ###
* [Robert Maier](https://vision.in.tum.de/members/maierr), Technische Universität München
* [Kihwan Kim](https://research.nvidia.com/person/kihwan-kim), NVIDIA
* [Daniel Cremers](https://vision.in.tum.de/members/cremers), Technische Universität München
* [Jan Kautz](https://research.nvidia.com/person/jan-kautz), NVIDIA
* [Matthias Nießner](http://www.niessnerlab.org/members/matthias_niessner/profile.html), Technische Universität München

## Summary
**Intrinsic3D** is a method to obtain high-quality 3D reconstructions from low-cost RGB-D sensors.
The algorithm recovers fine-scale geometric details and sharp surface textures by simultaneously optimizing for reconstructed geometry, surface albedos, camera poses and scene lighting.

This work is based on our publication  
* [Intrinsic3D: High-Quality 3D Reconstruction by Joint Appearance and Geometry Optimization with Spatially-Varying Lighting](https://vision.in.tum.de/_media/spezial/bib/maier2017intrinsic3d.pdf)  
International Conference on Computer Vision (ICCV) 2017.

If you find our source code or [dataset](https://vision.in.tum.de/data/datasets/intrinsic3d) useful in your research, please cite our work as follows:
```
@inproceedings{maier2017intrinsic3d,
   title = {{Intrinsic3D}: High-Quality {3D} Reconstruction by Joint Appearance and Geometry Optimization with Spatially-Varying Lighting},
   author = {Maier, Robert and Kim, Kihwan and Cremers, Daniel and Kautz, Jan and Nie{\ss}ner, Matthias},
   booktitle = {International Conference on Computer Vision (ICCV)},
   year = {2017}
}
```


## Installation
As the code was mostly developed and tested on Ubuntu Linux (16.10 and 18.10), we only provide the build instructions for Ubuntu in the following.
However, the code should also work on Windows with Visual Studio 2013.

Please first clone the source code:
```
git clone https://github.com/NVlabs/intrinsic3d.git
```

### Dependencies
Building Intrinsic3D requires
[CMake](https://cmake.org/download/),
[Eigen](http://eigen.tuxfamily.org/),
[OpenCV 4](https://opencv.org/releases.html),
[Boost](http://www.boost.org/users/download/) and
[Ceres Solver](http://ceres-solver.org/) (with [CXSparse](https://github.com/TheFrenchLeaf/CXSparse) on Windows)
as third-party libraries.
The following command installs the dependencies from the default Ubuntu repositories:
```
sudo apt install cmake libeigen3-dev libboost-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev libopencv-dev
```

Please install [Ceres Solver](http://ceres-solver.org/installation.html) using the following commands (if not installed already):
```
# create third_party subdirectory (in Intrinsic3D root folder)
mkdir third_party && cd third_party

# install Ceres Solver dependencies
sudo apt install libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev

# download and extract Ceres Solver source code package
wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
tar xvzf ceres-solver-2.1.0.tar.gz
cd ceres-solver-2.1.0

# compile and install Ceres Solver
mkdir build-ceres && cd build-ceres
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/../../ -DCXX11=ON -DSUITESPARSE=ON -DCXSPARSE=ON -DEIGENSPARSE=ON -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
make -j6
make install

# go back into source code root
cd ../../../
```

### Build
To compile Intrinsic3D, use the standard CMake approach:
```
mkdir build
cd build
cmake .. -DCeres_DIR=$PWD/../third_party/lib/cmake/Ceres
make -j6
```


## Data
Download one of the RGB-D sequences from the [Intrinsic3D Dataset](https://vision.in.tum.de/data/datasets/intrinsic3d). If you want to use one of your own datasets, you can convert it into the dataset format specified [here](https://vision.in.tum.de/data/datasets/intrinsic3d#format).
In the following, we will reconstruct and refine the [Lion sequence](https://vision.in.tum.de/_media/data/datasets/intrinsic3d/lion-rgbd.zip) using our approach.
```
# creating working folder in data/
cd ../data
mkdir lion && cd lion

# download, unzip and rename lion dataset
wget https://vision.in.tum.de/_media/data/datasets/intrinsic3d/lion-rgbd.zip
unzip lion-rgbd.zip
mv lion-rgbd rgbd
```


## Intrinsic3D Usage
To run the Intrinsic3D applications, we continue in the current data folder ```data/lion/``` and copy the default YAML configuration files into it:
```
# copy default configuration files into current data folder
cp ../*.yml .
```
The dataset configuration file ```sensor.yml``` is loaded in all applications and specifies the input RGB-D dataset.
Please note that the working folder of each application will be set to the folder containing the ```sensor.yml``` passed as argument.
The outputs will be generated in the subfolders ```fusion/``` and  ```intrinsic3d/``` respectively.

### Keyframe Selection
We first run the keyframe selection to discard blurry frames, based on a single-frame blurriness measure:
```
../../build/bin/AppKeyframes -s=sensor.yml -k=keyframes.yml
```
This will generate the keyframes file ```fusion/keyframes.txt```.
The window size for keyframe selection can be adjusted through the parameter ```window_size``` in the ```keyframes.yml``` configuration file.

### SDF Fusion
Next, the RGB-D input frames are fused in a voxel-hashed signed distance field (SDF), which produces an output 3D triangle mesh ```fusion/mesh_0.004.ply``` and an initial SDF volume
```fusion/volume_0.004.tsdf```:

```
../../build/bin/AppFusion -s=sensor.yml -f=fusion.yml
```
Since the framework is very computationally demanding, it is recommended to crop the 3D reconstruction already during the SDF fusion process.
You can therefore specify the 3D clip coordinates (in absolute coordinates of the first camera frame) by setting the ```crop_*``` parameters in the ```fusion.yml``` configuration file. 
To disable clipping, you can set all ```crop_*``` parameters to 0.0.
To accelerate finding a suitable clip volume, increase the ```voxel_size``` value and integrate keyframes only (by setting the parameter ```keyframes```).
For reference, here are suitable crop bounds for the [Intrinsic3D Dataset](https://vision.in.tum.de/data/datasets/intrinsic3d) sequences: 
```
Lion: left -0.09, right 1.55; top -0.58, bottom 0.26; front 0.0, back 2.0.
Gate: left -0.52, right 0.55; top -1.1, bottom 0.35; front 0.0, back 1.0.
Hieroglyphics: left -0.5, right 0.45; top -1.2, bottom 0.2; front 0.0, back 1.0.
Tomb Statuary: left -0.15, right 0.25; top -0.02, bottom 0.52; front 0.0, back 0.75.
Bricks: left -0.3, right 2.1; top -0.3, bottom 0.3; front 0.0, back 2.0.
```

### Intrinsic3D
The Intrinsic3D approach takes the initial SDF volume (```fusion/volume_0.004.tsdf```) and selected keyframes (```fusion/keyframes.txt```) and jointly optimizes the scene geometry and appearance along with the involved image formation model:

```
../../build/bin/AppIntrinsic3D -s=sensor.yml -i=intrinsic3d.yml
```

The output is generated in the subfolder ```intrinsic3d/```, with the final refined 3D mesh stored as ```mesh_g0_p0.ply```. The refined camera poses and color camera intrinsics are stored as ```poses_g0_p0.txt``` and ```intrinsics_g0_p0.txt``` respectively.
Intermediate results of coarser refinement levels are also output as
```mesh_g*_p*.ply```, where ```_g*``` specifies the SDF grid level and ```_p*``` stand for the RGB-D pyramid level (0 is always the finest/highest resolution).

The configuration file ```intrinsic3d.yml``` allows to adjust various parameters of the joint optimization method (e.g. hyperparameters for regularization terms).
In addition to only exporting the mesh colors, the ```output_mesh_*``` parameters enable other visualizations such as the refined albedo (set ```output_mesh_albedo``` to 1).

As the Intrinsic3D approach is computationally very demanding w.r.t. runtime and memory, it may require up to 32GB RAM during the optimization (even for object size reconstructions).
You can reduce the finest grid level resolution by decreasing the parameter ```num_grid_levels``` to reduce the memory usage. ```num_grid_levels: "3"``` means that the input SDF volume is upsampled twice; for an initial SDF grid with voxel size 0.004, the voxel size of the final refined SDF grid is 0.001.

## Contact
If you have any questions, please contact [Robert Maier &lt;robert.maier@tum.de>](mailto:robert.maier@tum.de) or [Kihwan Kim &lt;kihwank@nvidia.com>](mailto:kihwank@nvidia.com).
